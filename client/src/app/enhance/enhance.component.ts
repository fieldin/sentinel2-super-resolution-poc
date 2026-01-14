import { Component, signal, computed, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { EsrganService, EnhanceProgress, EnhanceResponse } from '../services/esrgan.service';
import { QrOcrService, OcrResult } from '../services/qr-ocr.service';

export interface SRModel {
  id: string;
  name: string;
  description: string;
  scale: number;
  bestFor: string;
}

@Component({
  selector: 'app-enhance',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './enhance.component.html',
  styleUrls: ['./enhance.component.scss']
})
export class EnhanceComponent implements OnDestroy {
  // Available SR models
  availableModels: SRModel[] = [
    {
      id: 'realesrgan_x4',
      name: 'Real-ESRGAN x4',
      description: 'Best quality for photos & landscapes',
      scale: 4,
      bestFor: 'General photos, landscapes, portraits'
    },
    {
      id: 'realesrgan_anime',
      name: 'Real-ESRGAN Anime 6B',
      description: 'Sharp edges, optimized for text/graphics',
      scale: 4,
      bestFor: 'Text, license plates, graphics, anime'
    }
  ];

  // State signals
  isDragging = signal(false);
  selectedFile = signal<File | null>(null);
  previewUrl = signal<string | null>(null);
  enhancedUrl = signal<string | null>(null);
  isProcessing = signal(false);
  result = signal<EnhanceResponse | null>(null);
  error = signal<string | null>(null);
  selectedModel = signal<string>('realesrgan_x4');
  progress = signal<EnhanceProgress>({
    stage: 'uploading',
    progress: 0,
    message: 'Ready to upload'
  });

  // QR/OCR state signals
  isReadingQrOcr = signal(false);
  qrOcrStatus = signal<string>('');
  extractedValue = signal<string | null>(null);
  extractedSource = signal<'qr' | 'ocr' | null>(null); // Track if result came from QR or OCR
  copyFeedback = signal<string | null>(null);

  // Computed values
  hasImage = computed(() => !!this.selectedFile());
  canProcess = computed(() => this.hasImage() && !this.isProcessing());

  progressPercent = computed(() => {
    const p = this.progress();
    // Combined progress: upload (0-30%), processing (30-90%), download (90-100%)
    switch (p.stage) {
      case 'uploading':
        return Math.round(p.progress * 0.3);
      case 'processing':
        return 30 + Math.round(p.progress * 0.6);
      case 'downloading':
        return 90 + Math.round(p.progress * 0.1);
      case 'complete':
        return 100;
      default:
        return 0;
    }
  });

  stageLabel = computed(() => {
    const p = this.progress();
    switch (p.stage) {
      case 'uploading':
        return '📤 Uploading';
      case 'processing':
        return '🔮 Enhancing';
      case 'downloading':
        return '📥 Downloading';
      case 'complete':
        return '✅ Complete';
      case 'error':
        return '❌ Error';
      default:
        return '';
    }
  });

  constructor(
    private esrganService: EsrganService,
    private qrOcrService: QrOcrService
  ) {
    // Subscribe to progress updates
    this.esrganService.progress$.subscribe(progress => {
      this.progress.set(progress);
    });
  }

  ngOnDestroy(): void {
    // Cleanup QR/OCR service resources
    this.qrOcrService.destroy();
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(false);

    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.handleFile(files[0]);
    }
  }

  onFileSelect(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.handleFile(input.files[0]);
    }
  }

  private handleFile(file: File): void {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      this.error.set('Please select a valid image file');
      return;
    }

    // Validate file size (max 50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      this.error.set('File size must be less than 50MB');
      return;
    }

    this.error.set(null);
    this.selectedFile.set(file);
    this.enhancedUrl.set(null);
    this.result.set(null);
    this.esrganService.reset();

    // Create preview URL
    const reader = new FileReader();
    reader.onload = () => {
      this.previewUrl.set(reader.result as string);
    };
    reader.readAsDataURL(file);
  }

  onModelChange(modelId: string): void {
    this.selectedModel.set(modelId);
  }

  getSelectedModelInfo(): SRModel | undefined {
    return this.availableModels.find(m => m.id === this.selectedModel());
  }

  processImage(): void {
    const file = this.selectedFile();
    if (!file) return;

    this.isProcessing.set(true);
    this.error.set(null);
    this.enhancedUrl.set(null);

    this.esrganService.enhanceImage(file, this.selectedModel()).subscribe({
      next: (response) => {
        this.isProcessing.set(false);
        this.result.set(response);
        this.enhancedUrl.set(response.download_url);

        // Auto-download the enhanced image after processing completes
        setTimeout(() => {
          this.downloadEnhanced();
        }, 500);
      },
      error: (err) => {
        this.isProcessing.set(false);
        const errorMsg = err?.message || err?.error?.message || 'Failed to enhance image. Please try again.';
        this.error.set(errorMsg);
        this.esrganService.setError(errorMsg);
      }
    });
  }

  downloadEnhanced(): void {
    const url = this.enhancedUrl();
    const result = this.result();
    if (!url || !result) return;

    const link = document.createElement('a');
    link.href = url;
    link.download = result.enhanced_filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  reset(): void {
    this.selectedFile.set(null);
    this.previewUrl.set(null);
    this.enhancedUrl.set(null);
    this.result.set(null);
    this.error.set(null);
    this.isProcessing.set(false);
    this.esrganService.reset();
    // Also reset QR/OCR state
    this.resetQrOcr();
  }

  formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  // ============================================
  // QR/OCR Feature Methods
  // ============================================

  /**
   * Main orchestration function for QR/OCR reading
   * Flow: QR decode → OCR on original → Enhance + OCR retry
   */
  async handleReadQrOrOcr(): Promise<void> {
    const file = this.selectedFile();
    const previewDataUrl = this.previewUrl();
    
    if (!file && !previewDataUrl) {
      this.qrOcrStatus.set('❌ No image loaded');
      return;
    }

    // Use the enhanced image if available, otherwise the original
    const imageSource = this.enhancedUrl() || previewDataUrl;
    if (!imageSource) {
      this.qrOcrStatus.set('❌ No image available');
      return;
    }

    this.isReadingQrOcr.set(true);
    this.extractedValue.set(null);
    this.extractedSource.set(null);
    this.copyFeedback.set(null);

    try {
      // Step 1: Try QR code first (fast)
      this.qrOcrStatus.set('🔍 Decoding QR code...');
      console.log('[Enhance] Step 1: Attempting QR decode...');
      
      const qrResult = await this.qrOcrService.decodeQrFromImage(imageSource);
      
      if (qrResult) {
        // QR success!
        this.extractedValue.set(qrResult);
        this.extractedSource.set('qr');
        this.qrOcrStatus.set('✅ QR code decoded successfully!');
        console.log('[Enhance] QR decode success:', qrResult);
        this.isReadingQrOcr.set(false);
        return;
      }

      // Step 2: QR failed, try OCR on current image
      this.qrOcrStatus.set('🔠 QR not found, running OCR...');
      console.log('[Enhance] Step 2: QR not found, running OCR on current image...');
      
      let ocrResult = await this.qrOcrService.runOcr(imageSource);
      let serial = this.qrOcrService.extractSerial(ocrResult.text, ocrResult.words);

      // Check if we got a good result
      const needsEnhancement = this.qrOcrService.shouldRetryWithEnhancement(ocrResult, serial);

      if (!needsEnhancement && serial) {
        // Good OCR result!
        this.extractedValue.set(serial);
        this.extractedSource.set('ocr');
        this.qrOcrStatus.set(`✅ Serial extracted (confidence: ${Math.round(ocrResult.confidence)}%)`);
        console.log('[Enhance] OCR success:', serial, 'confidence:', ocrResult.confidence);
        this.isReadingQrOcr.set(false);
        return;
      }

      // Step 3: OCR quality low, enhance image and retry
      // Only do this if we haven't already enhanced the image
      if (!this.enhancedUrl()) {
        this.qrOcrStatus.set('🔮 Enhancing image for better OCR...');
        console.log('[Enhance] Step 3: Enhancing image with Anime 6B model...');

        try {
          // Use the Anime 6B model which is best for text/plates
          const enhanceResult = await this.enhanceForOcr(file!);
          
          if (enhanceResult) {
            // Run OCR on enhanced image
            this.qrOcrStatus.set('🔠 Running OCR on enhanced image...');
            console.log('[Enhance] Running OCR on enhanced image...');
            
            ocrResult = await this.qrOcrService.runOcr(enhanceResult);
            serial = this.qrOcrService.extractSerial(ocrResult.text, ocrResult.words);

            if (serial) {
              this.extractedValue.set(serial);
              this.extractedSource.set('ocr');
              this.qrOcrStatus.set(`✅ Serial extracted from enhanced image (confidence: ${Math.round(ocrResult.confidence)}%)`);
              console.log('[Enhance] Enhanced OCR success:', serial);
              this.isReadingQrOcr.set(false);
              return;
            }
          }
        } catch (enhanceError) {
          console.error('[Enhance] Enhancement failed:', enhanceError);
          // Continue with original OCR result if enhancement fails
        }
      }

      // If we still have a serial from original OCR, use it
      if (serial) {
        this.extractedValue.set(serial);
        this.extractedSource.set('ocr');
        this.qrOcrStatus.set(`⚠️ Serial extracted (low confidence: ${Math.round(ocrResult.confidence)}%)`);
        console.log('[Enhance] Using low-confidence OCR result:', serial);
      } else {
        this.qrOcrStatus.set('❌ No serial/PIN found in image');
        console.log('[Enhance] No serial found after all attempts');
      }

    } catch (error) {
      console.error('[Enhance] QR/OCR error:', error);
      this.qrOcrStatus.set(`❌ Error: ${(error as Error).message || 'Unknown error'}`);
    } finally {
      this.isReadingQrOcr.set(false);
    }
  }

  /**
   * Enhance image specifically for OCR using the Anime 6B model
   * Returns the enhanced image URL
   */
  private enhanceForOcr(file: File): Promise<string | null> {
    return new Promise((resolve, reject) => {
      // Use the anime model which is best for text/plates
      const modelId = 'realesrgan_anime';

      this.esrganService.enhanceImage(file, modelId).subscribe({
        next: (response) => {
          console.log('[Enhance] Enhancement for OCR complete');
          this.enhancedUrl.set(response.download_url);
          this.result.set(response);
          resolve(response.download_url);
        },
        error: (err) => {
          console.error('[Enhance] Enhancement for OCR failed:', err);
          reject(err);
        }
      });
    });
  }

  /**
   * Copy extracted value to clipboard
   */
  async copyToClipboard(): Promise<void> {
    const value = this.extractedValue();
    if (!value) return;

    try {
      await navigator.clipboard.writeText(value);
      this.copyFeedback.set('Copied!');
      console.log('[Enhance] Copied to clipboard:', value);
      
      // Clear feedback after 2 seconds
      setTimeout(() => {
        this.copyFeedback.set(null);
      }, 2000);
    } catch (error) {
      console.error('[Enhance] Clipboard write failed:', error);
      this.copyFeedback.set('Copy failed');
      
      // Fallback: select the text (if using an input field)
      setTimeout(() => {
        this.copyFeedback.set(null);
      }, 2000);
    }
  }

  /**
   * Reset QR/OCR state
   */
  resetQrOcr(): void {
    this.extractedValue.set(null);
    this.extractedSource.set(null);
    this.qrOcrStatus.set('');
    this.copyFeedback.set(null);
  }
}

