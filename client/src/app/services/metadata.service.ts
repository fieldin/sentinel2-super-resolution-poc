import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

export interface TilesetMetadata {
  bounds: [number, number, number, number]; // [west, south, east, north]
  minzoom: number;
  maxzoom: number;
  tileTemplate: string;
  attribution: string;
  format: string;
  tileSize: number;
}

export interface SourceMetadata {
  acquisition_date: string;
  scene_id: string;
  cloud_cover_pct: number;
  crs: string;
  bbox: number[];
  job_id: string | null;
  file_path: string;
  file_size_mb: number;
  downloaded_at: string;
  source: string;
  is_mock?: boolean;
}

export interface MetadataResponse {
  tileset: TilesetMetadata | null;
  source: SourceMetadata | null;
  tilesAvailable: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class MetadataService {
  constructor(private http: HttpClient) {}

  getMetadata(): Observable<MetadataResponse> {
    return this.http.get<MetadataResponse>('/api/metadata').pipe(
      tap(meta => console.log('Metadata loaded:', meta))
    );
  }
}

