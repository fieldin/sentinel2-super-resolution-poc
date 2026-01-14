import { Injectable } from '@angular/core';
import { HttpClient, HttpEvent, HttpEventType, HttpProgressEvent } from '@angular/common/http';
import { Observable, Subject, BehaviorSubject, interval } from 'rxjs';
import { map, tap, filter, switchMap, take } from 'rxjs/operators';

export interface EnhanceProgress {
  stage: 'uploading' | 'processing' | 'downloading' | 'complete' | 'error';
  progress: number; // 0-100
  message: string;
}

export interface EnhanceResponse {
  success: boolean;
  message?: string;
  original_filename: string;
  enhanced_filename: string;
  original_size: { width: number; height: number };
  enhanced_size: { width: number; height: number };
  scale_factor: number;
  processing_time_seconds: number;
  download_url: string;
}

@Injectable({
  providedIn: 'root'
})
export class EsrganService {
  private progressSubject = new BehaviorSubject<EnhanceProgress>({
    stage: 'uploading',
    progress: 0,
    message: 'Ready to upload'
  });

  progress$ = this.progressSubject.asObservable();

  constructor(private http: HttpClient) {}

  /**
   * Upload an image for Real-ESRGAN enhancement
   * Returns observable that emits progress updates and final result
   * @param file - The image file to enhance
   * @param model - The SR model to use (default: realesrgan_x4)
   */
  enhanceImage(file: File, model: string = 'realesrgan_x4'): Observable<EnhanceResponse> {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model', model);

    // Reset progress
    this.progressSubject.next({
      stage: 'uploading',
      progress: 0,
      message: 'Uploading image...'
    });

    // Upload and get job_id, then poll /api/sr/{job_id} for status
    return new Observable<EnhanceResponse>((observer) => {
      this.http.post<any>('/api/enhance', formData, {
        reportProgress: true,
        observe: 'events'
      }).subscribe({
        next: (event: HttpEvent<any>) => {
          // Handle upload progress events
          if (event.type === HttpEventType.UploadProgress) {
            const uploadProgress = event as HttpProgressEvent;
            if (uploadProgress.total) {
              const percent = Math.round(100 * uploadProgress.loaded / uploadProgress.total);
              this.progressSubject.next({
                stage: 'uploading',
                progress: percent,
                message: `Uploading image... ${percent}%`
              });
            }
          }

          // When upload finishes, server returns JSON with job_id
          if (event.type === HttpEventType.Response) {
            const body = (event as any).body || {};
            const jobId = body.job_id || body.jobId || body.jobId;
            if (!jobId) {
              observer.error(new Error('No job_id returned from server'));
              return;
            }

            // Switch to processing stage
            this.progressSubject.next({
              stage: 'processing',
              progress: 0,
              message: 'Queued for processing...'
            });

            // Poll loop
            const poll = () => {
              this.http.get<any>(`/api/sr/${jobId}`).subscribe({
                next: (status) => {
                  const s = status.status || '';
                  if (s === 'completed') {
                    // Map server result to EnhanceResponse
                    const result = status.result || {};
                    const sr_meta = result['sr_metadata'] || {};
                    const outputs = result['outputs'] || {};
                    const out_file = outputs['sr_png'] || outputs['sr_tif'] || '';
                    // compute processing time if available
                    let processing_seconds = 0;
                    try {
                      if (status.created_at && status.completed_at) {
                        const start = new Date(status.created_at).getTime();
                        const end = new Date(status.completed_at).getTime();
                        processing_seconds = (end - start) / 1000;
                      }
                    } catch (e) {}

                    const resp: EnhanceResponse = {
                      success: true,
                      message: status.message || 'Enhancement complete',
                      original_filename: (status.input_file || '').split('/').pop() || '',
                      enhanced_filename: (out_file || '').split('/').pop() || '',
                      original_size: {
                        width: (sr_meta['original_size'] || [0,0])[1],
                        height: (sr_meta['original_size'] || [0,0])[0]
                      },
                      enhanced_size: {
                        width: (sr_meta['output_size'] || [0,0])[1],
                        height: (sr_meta['output_size'] || [0,0])[0]
                      },
                      scale_factor: sr_meta['scale'] || 4,
                      processing_time_seconds: processing_seconds,
                      download_url: `/api/sr/download/${jobId}/${(out_file || '').split('/').pop() || ''}`
                    };

                    this.progressSubject.next({
                      stage: 'complete',
                      progress: 100,
                      message: resp.message || 'Enhancement complete'
                    });

                    observer.next(resp);
                    observer.complete();
                    return;
                  } else if (s === 'failed') {
                    observer.error(new Error(status.message || 'Enhancement failed'));
                    return;
                  } else {
                    // Parse progress from message like "Stage 1/2: ..." or "Stage 2/2: ..."
                    let progress = 0;
                    const msg = status.message || 'Processing...';
                    const stageMatch = msg.match(/Stage\s*(\d+)\s*\/\s*(\d+)/i);
                    if (stageMatch) {
                      const current = parseInt(stageMatch[1], 10);
                      const total = parseInt(stageMatch[2], 10);
                      // Each stage takes equal portion, calculate mid-point of current stage
                      progress = Math.round(((current - 0.5) / total) * 100);
                    } else if (s === 'queued') {
                      progress = 5;
                    } else {
                      progress = 50; // Default if can't parse
                    }
                    
                    // still queued/processing - update progress message
                    this.progressSubject.next({
                      stage: 'processing',
                      progress: progress,
                      message: msg
                    });
                    // schedule next poll
                    setTimeout(poll, 2000);
                  }
                },
                error: (err) => observer.error(err)
              });
            };

            // start polling
            setTimeout(poll, 500);
          }
        },
        error: (err) => {
          this.setError(err.error?.message || 'Upload failed');
          observer.error(err);
        }
      });
    });
  }

  private handleProgress(event: HttpEvent<any>): void {
    switch (event.type) {
      case HttpEventType.UploadProgress:
        const uploadProgress = event as HttpProgressEvent;
        if (uploadProgress.total) {
          const percent = Math.round(100 * uploadProgress.loaded / uploadProgress.total);
          this.progressSubject.next({
            stage: 'uploading',
            progress: percent,
            message: `Uploading image... ${percent}%`
          });
          
          // When upload completes, switch to processing stage
          if (percent === 100) {
            this.progressSubject.next({
              stage: 'processing',
              progress: 0,
              message: 'Processing with Real-ESRGAN...'
            });
            
            // Simulate processing progress (server doesn't send progress during processing)
            this.simulateProcessingProgress();
          }
        }
        break;
      
      case HttpEventType.DownloadProgress:
        const downloadProgress = event as HttpProgressEvent;
        if (downloadProgress.total) {
          const percent = Math.round(100 * downloadProgress.loaded / downloadProgress.total);
          this.progressSubject.next({
            stage: 'downloading',
            progress: percent,
            message: `Receiving result... ${percent}%`
          });
        }
        break;
    }
  }

  private simulateProcessingProgress(): void {
    let progress = 0;
    const interval = setInterval(() => {
      const current = this.progressSubject.getValue();
      
      // Stop if we've moved past processing stage
      if (current.stage !== 'processing') {
        clearInterval(interval);
        return;
      }
      
      // Slowly increment progress to simulate work
      progress += Math.random() * 5;
      if (progress > 95) progress = 95; // Cap at 95% until real completion
      
      this.progressSubject.next({
        stage: 'processing',
        progress: Math.round(progress),
        message: `Processing with Real-ESRGAN... ${Math.round(progress)}%`
      });
    }, 500);
  }

  /**
   * Report an error
   */
  setError(message: string): void {
    this.progressSubject.next({
      stage: 'error',
      progress: 0,
      message
    });
  }

  /**
   * Reset the service state
   */
  reset(): void {
    this.progressSubject.next({
      stage: 'uploading',
      progress: 0,
      message: 'Ready to upload'
    });
  }
}

