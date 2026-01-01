import {
  Component,
  OnInit,
  OnDestroy,
  ElementRef,
  ViewChild,
  AfterViewInit,
  signal,
  computed
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import mapboxgl from 'mapbox-gl';

import { ConfigService, AppConfig } from '../services/config.service';
import { MetadataService, MetadataResponse, SourceMetadata } from '../services/metadata.service';

@Component({
  selector: 'app-map',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.scss']
})
export class MapComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('mapContainer', { static: true }) mapContainer!: ElementRef<HTMLDivElement>;

  private map: mapboxgl.Map | null = null;

  // Reactive state
  loading = signal(true);
  error = signal<string | null>(null);
  tilesAvailable = signal(false);
  
  opacity = signal(80);
  currentZoom = signal(12);
  
  sourceInfo = signal<SourceMetadata | null>(null);
  
  // Computed values
  formattedAcquisitionDate = computed(() => {
    const source = this.sourceInfo();
    if (!source?.acquisition_date) return 'N/A';
    return new Date(source.acquisition_date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  });

  constructor(
    private configService: ConfigService,
    private metadataService: MetadataService
  ) {}

  ngOnInit(): void {
    this.loadData();
  }

  ngAfterViewInit(): void {
    // Map initialization happens after data is loaded
  }

  ngOnDestroy(): void {
    this.map?.remove();
  }

  private loadData(): void {
    forkJoin({
      config: this.configService.getConfig(),
      metadata: this.metadataService.getMetadata()
    }).subscribe({
      next: ({ config, metadata }) => {
        this.initializeMap(config, metadata);
      },
      error: (err) => {
        console.error('Failed to load data:', err);
        this.error.set('Failed to load configuration. Is the server running?');
        this.loading.set(false);
      }
    });
  }

  private initializeMap(config: AppConfig, metadata: MetadataResponse): void {
    // Set Mapbox access token
    mapboxgl.accessToken = config.mapboxAccessToken;

    // Create map
    this.map = new mapboxgl.Map({
      container: this.mapContainer.nativeElement,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: config.defaultCenter,
      zoom: config.defaultZoom,
      minZoom: config.tileMinZoom - 2,
      maxZoom: config.tileMaxZoom + 2
    });

    // Add navigation controls
    this.map.addControl(new mapboxgl.NavigationControl(), 'top-right');
    this.map.addControl(new mapboxgl.ScaleControl(), 'bottom-right');

    // Map load handler
    this.map.on('load', () => {
      this.loading.set(false);
      this.tilesAvailable.set(metadata.tilesAvailable);
      this.sourceInfo.set(metadata.source);

      if (metadata.tilesAvailable && metadata.tileset) {
        this.addSentinelLayer(metadata);
      }
    });

    // Track zoom level
    this.map.on('zoom', () => {
      this.currentZoom.set(Math.round(this.map!.getZoom() * 10) / 10);
    });

    // Handle errors
    this.map.on('error', (e) => {
      console.error('Map error:', e);
    });
  }

  private addSentinelLayer(metadata: MetadataResponse): void {
    if (!this.map || !metadata.tileset) return;

    const tileset = metadata.tileset;
    const tileUrl = window.location.origin + tileset.tileTemplate;

    // Add raster source
    this.map.addSource('sentinel-tiles', {
      type: 'raster',
      tiles: [tileUrl],
      tileSize: tileset.tileSize || 256,
      bounds: tileset.bounds,
      minzoom: tileset.minzoom,
      maxzoom: tileset.maxzoom,
      attribution: tileset.attribution
    });

    // Add raster layer
    this.map.addLayer({
      id: 'sentinel-layer',
      type: 'raster',
      source: 'sentinel-tiles',
      paint: {
        'raster-opacity': this.opacity() / 100,
        'raster-fade-duration': 0
      }
    });

    // Fit to bounds
    const bounds = tileset.bounds;
    this.map.fitBounds(
      [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
      { padding: 50, duration: 1000 }
    );
  }

  onOpacityChange(value: number): void {
    this.opacity.set(value);
    if (this.map?.getLayer('sentinel-layer')) {
      this.map.setPaintProperty('sentinel-layer', 'raster-opacity', value / 100);
    }
  }

  zoomToExtent(): void {
    const source = this.sourceInfo();
    if (!this.map || !source?.bbox) return;

    const bbox = source.bbox;
    this.map.fitBounds(
      [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
      { padding: 50, duration: 1000 }
    );
  }

  refreshData(): void {
    this.loading.set(true);
    this.error.set(null);
    
    // Remove existing layer and source
    if (this.map) {
      if (this.map.getLayer('sentinel-layer')) {
        this.map.removeLayer('sentinel-layer');
      }
      if (this.map.getSource('sentinel-tiles')) {
        this.map.removeSource('sentinel-tiles');
      }
    }

    // Reload metadata
    this.metadataService.getMetadata().subscribe({
      next: (metadata) => {
        this.loading.set(false);
        this.tilesAvailable.set(metadata.tilesAvailable);
        this.sourceInfo.set(metadata.source);

        if (metadata.tilesAvailable && metadata.tileset) {
          this.addSentinelLayer(metadata);
        }
      },
      error: (err) => {
        console.error('Failed to refresh:', err);
        this.error.set('Failed to refresh data');
        this.loading.set(false);
      }
    });
  }
}

