import { Injectable } from '@angular/core';
import jsQR from 'jsqr';
import Tesseract from 'tesseract.js';

export interface OcrResult {
  text: string;
  confidence: number;
  words: Array<{ text: string; confidence: number }>;
}

export interface SerialCandidate {
  text: string;
  confidence: number;
  length: number;
}

/**
 * Service for QR code decoding and OCR text extraction
 * Used for reading VIN/Serial numbers from tractor plates
 */
@Injectable({
  providedIn: 'root'
})
export class QrOcrService {
  private tesseractWorker: Tesseract.Worker | null = null;
  private workerReady = false;

  constructor() {}

  /**
   * Initialize Tesseract worker for OCR
   * Called lazily on first OCR request
   */
  private async initTesseract(): Promise<void> {
    if (this.workerReady && this.tesseractWorker) return;

    console.log('[QrOcrService] Initializing Tesseract worker...');
    this.tesseractWorker = await Tesseract.createWorker('eng', 1, {
      logger: (m) => {
        if (m.status === 'recognizing text') {
          console.log(`[OCR] Progress: ${Math.round(m.progress * 100)}%`);
        }
      }
    });
    this.workerReady = true;
    console.log('[QrOcrService] Tesseract worker ready');
  }

  /**
   * Load an image from various sources and return ImageData for processing
   */
  private async getImageData(source: File | Blob | string): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';

      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        resolve(imageData);
      };

      img.onerror = () => reject(new Error('Failed to load image'));

      if (source instanceof File || source instanceof Blob) {
        img.src = URL.createObjectURL(source);
      } else {
        // Assume it's a data URL or regular URL
        img.src = source;
      }
    });
  }

  /**
   * Decode QR code from an image
   * @param source - File, Blob, or data URL of the image
   * @returns Decoded QR text or null if no QR code found
   */
  async decodeQrFromImage(source: File | Blob | string): Promise<string | null> {
    console.log('[QrOcrService] Attempting QR decode...');
    try {
      const imageData = await this.getImageData(source);
      const code = jsQR(imageData.data, imageData.width, imageData.height, {
        inversionAttempts: 'attemptBoth'
      });

      if (code && code.data) {
        console.log('[QrOcrService] QR decode success:', code.data);
        return code.data;
      }

      console.log('[QrOcrService] No QR code found in image');
      return null;
    } catch (error) {
      console.error('[QrOcrService] QR decode error:', error);
      return null;
    }
  }

  /**
   * Run OCR on an image to extract text
   * @param source - File, Blob, or data URL of the image
   * @returns OCR result with text and confidence
   */
  async runOcr(source: File | Blob | string): Promise<OcrResult> {
    console.log('[QrOcrService] Running OCR...');
    await this.initTesseract();

    if (!this.tesseractWorker) {
      throw new Error('Tesseract worker not initialized');
    }

    try {
      // Convert source to a format Tesseract can handle
      let imageSrc: string;
      if (source instanceof File || source instanceof Blob) {
        imageSrc = URL.createObjectURL(source);
      } else {
        imageSrc = source;
      }

      const result = await this.tesseractWorker.recognize(imageSrc);
      
      // Extract word-level confidence
      const words = result.data.words?.map(w => ({
        text: w.text,
        confidence: w.confidence
      })) || [];

      const ocrResult: OcrResult = {
        text: result.data.text,
        confidence: result.data.confidence,
        words
      };

      console.log('[QrOcrService] OCR complete. Confidence:', ocrResult.confidence);
      console.log('[QrOcrService] Raw text:', ocrResult.text.substring(0, 200));

      return ocrResult;
    } catch (error) {
      console.error('[QrOcrService] OCR error:', error);
      throw error;
    }
  }

  /**
   * Extract the best serial/PIN candidate from OCR text
   * Looks for patterns typical of Kubota PINs and other tractor serials
   * @param text - Raw OCR text
   * @param words - Word-level OCR results with confidence
   * @returns Best matching serial string or null
   */
  extractSerial(text: string, words?: Array<{ text: string; confidence: number }>): string | null {
    console.log('[QrOcrService] Extracting serial from text...');

    // Normalize: uppercase, remove common OCR artifacts
    const normalized = text
      .toUpperCase()
      .replace(/[^A-Z0-9\s\n-]/g, '') // Keep only alphanumeric, spaces, newlines, hyphens
      .replace(/\s+/g, ' ') // Collapse whitespace
      .trim();

    console.log('[QrOcrService] Normalized text:', normalized.substring(0, 200));

    // Serial/PIN patterns to look for:
    // - Kubota PINs: typically 17 chars (like vehicle VINs)
    // - Equipment serials: 10-20 alphanumeric chars
    // - May contain hyphens
    
    // Pattern: 10-20 alphanumeric characters (with optional hyphens)
    const serialPattern = /[A-Z0-9][A-Z0-9-]{8,18}[A-Z0-9]/g;
    const matches = normalized.match(serialPattern) || [];

    // Also try without hyphens for cleaner candidates
    const alphanumericPattern = /[A-Z0-9]{10,20}/g;
    const alphaMatches = normalized.replace(/-/g, '').match(alphanumericPattern) || [];

    // Combine and dedupe candidates
    const allCandidates = [...new Set([...matches, ...alphaMatches])];

    console.log('[QrOcrService] Found candidates:', allCandidates);

    if (allCandidates.length === 0) {
      console.log('[QrOcrService] No serial candidates found');
      return null;
    }

    // Score candidates based on:
    // 1. Length (prefer 14-17 chars like VINs)
    // 2. Word confidence if available
    // 3. Pattern quality (mix of letters and numbers is better)
    const scoredCandidates: SerialCandidate[] = allCandidates.map(candidate => {
      let score = 0;
      const cleanCandidate = candidate.replace(/-/g, '');
      const len = cleanCandidate.length;

      // Length scoring: prefer 14-17 chars (VIN length)
      if (len >= 14 && len <= 17) {
        score += 50;
      } else if (len >= 10 && len <= 20) {
        score += 30;
      }

      // Pattern quality: good serials have mix of letters and numbers
      const hasLetters = /[A-Z]/.test(cleanCandidate);
      const hasNumbers = /[0-9]/.test(cleanCandidate);
      if (hasLetters && hasNumbers) {
        score += 20;
      }

      // Penalize if it looks like a date or simple number
      if (/^\d+$/.test(cleanCandidate)) {
        score -= 20;
      }

      // Word confidence boost if available
      if (words && words.length > 0) {
        const matchingWords = words.filter(w => 
          candidate.includes(w.text.toUpperCase()) || 
          w.text.toUpperCase().includes(candidate)
        );
        if (matchingWords.length > 0) {
          const avgConf = matchingWords.reduce((sum, w) => sum + w.confidence, 0) / matchingWords.length;
          score += avgConf / 10; // Add up to 10 points based on confidence
        }
      }

      return {
        text: candidate,
        confidence: score,
        length: len
      };
    });

    // Sort by score descending
    scoredCandidates.sort((a, b) => b.confidence - a.confidence);

    console.log('[QrOcrService] Scored candidates:', scoredCandidates);

    const best = scoredCandidates[0];
    if (best) {
      console.log('[QrOcrService] Best serial candidate:', best.text, 'score:', best.confidence);
      return best.text;
    }

    return null;
  }

  /**
   * Check if OCR result indicates low confidence or no serial found
   * Used to determine if we should enhance and retry
   */
  shouldRetryWithEnhancement(ocrResult: OcrResult, extractedSerial: string | null): boolean {
    // Retry if:
    // 1. No serial was extracted
    // 2. Overall OCR confidence is below 60%
    // 3. The extracted serial is suspiciously short
    
    if (!extractedSerial) {
      console.log('[QrOcrService] Should retry: no serial found');
      return true;
    }

    if (ocrResult.confidence < 60) {
      console.log('[QrOcrService] Should retry: low confidence', ocrResult.confidence);
      return true;
    }

    const cleanSerial = extractedSerial.replace(/-/g, '');
    if (cleanSerial.length < 10) {
      console.log('[QrOcrService] Should retry: serial too short', cleanSerial.length);
      return true;
    }

    return false;
  }

  /**
   * Cleanup resources
   */
  async destroy(): Promise<void> {
    if (this.tesseractWorker) {
      await this.tesseractWorker.terminate();
      this.tesseractWorker = null;
      this.workerReady = false;
    }
  }
}

