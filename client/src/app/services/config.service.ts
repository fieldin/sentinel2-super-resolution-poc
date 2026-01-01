import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, shareReplay, tap } from 'rxjs';

export interface AppConfig {
  mapboxAccessToken: string;
  tileMinZoom: number;
  tileMaxZoom: number;
  defaultCenter: [number, number];
  defaultZoom: number;
}

@Injectable({
  providedIn: 'root'
})
export class ConfigService {
  private config$: Observable<AppConfig> | null = null;

  constructor(private http: HttpClient) {}

  getConfig(): Observable<AppConfig> {
    if (!this.config$) {
      this.config$ = this.http.get<AppConfig>('/api/config').pipe(
        tap(config => console.log('Config loaded:', config)),
        shareReplay(1)
      );
    }
    return this.config$;
  }
}

