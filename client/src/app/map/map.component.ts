import {
  Component,
  OnInit,
  OnDestroy,
  signal,
  computed
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import * as _ from 'lodash';
import L from 'leaflet';

import { FldMapModule } from 'fieldin-web-20/fld-map';
import { ConfigService, AppConfig } from '../services/config.service';
import { MetadataService, MetadataResponse, SourceMetadata, TilesetMetadata } from '../services/metadata.service';

// User location configuration
const USER_LOCATION = {
  lat: 36.6229233241847,
  lng: -121.67341900517911
};

@Component({
  selector: 'app-map',
  standalone: true,
  imports: [CommonModule, FormsModule, FldMapModule],
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.scss']
})
export class MapComponent implements OnInit, OnDestroy {
  // Reactive state
  loading = signal(true);
  error = signal<string | null>(null);
  tilesAvailable = signal(false);
  wowTilesAvailable = signal(false);
  isMapReady = signal(false);

  opacity = signal(80);
  currentZoom = signal(12);
  useWowTiles = signal(true);
  drawnPolygons = signal<any[]>([]);

  sourceInfo = signal<SourceMetadata | null>(null);

  // Map configuration for fld-map
  mapConfiguration: any;
  private mapServices: any = {};
  private userLocationMarker: L.Marker | null = null;
  private srTileLayer: L.TileLayer | null = null;
  private config: AppConfig | null = null;
  private tileset: TilesetMetadata | null = null;
  private tileEndpoints: any = null;

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
  ) { }

  ngOnInit(): void {
    this.initMapConfiguration();
    this.loadData();
  }

  private initMapConfiguration(): void {
    // Initialize map configuration with polygon editing enabled
    // Following the storybook pattern exactly
    this.mapConfiguration = {
      services: this.mapServices,
      polygonManagement: false,       // Don't block clicks on polygons
      editPolygons: true,             // Enable polygon editing mode
      onlySelectOne: false,           // Allow continuous editing
      defaultMap: 'googleMap',
      mapboxStyle: 'mapbox://styles/mapbox/satellite-streets-v11',
      // Map keys from fld-app-new environment
      mapKeys: {
        mapboxKey: 'pk.eyJ1IjoiZHJvcmUiLCJhIjoiYkxSbmZYUSJ9.Ij81_3Tc3HFJTkrzmzIUBw',
        googleMapsKey: 'AIzaSyCMPZFZXNTZkYE12DXQ9L06Dw095hb9xhk',
        appleMapsKey: 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkxGWDQ5V0tIWlEifQ.eyJpYXQiOjE3NDUzMTI2ODcsImV4cCI6MjA2MDY3MjY4NywiaXNzIjoiRzMzM1FEQUFLNCJ9.W0i3hY3OSSMuJTbmVaSjgTUlgJYlebV_d8doS7viZi5QodjazUufkoVcS3dYpr7bQwRrjJ5g1-OqNEGEtqIxfw'
      },
      initialState: {
        lng: USER_LOCATION.lng,
        lat: USER_LOCATION.lat,
        zoom: 12,
        maxZoom: 25
      },
      mapEvents: (event: any) => this.handleMapEvents(event),
      // Custom color for polygons
      getPolygonColorByType: (d: any) => {
        if (d.polygonData?.type === 'field') {
          return [255, 165, 0]; // Orange for fields
        } else if (d.polygonData?.type === 'zone') {
          return [0, 128, 255]; // Blue for zones
        }
        return [0, 212, 170]; // Default accent color
      },
      getPolygonFillByType: (d: any) => {
        if (d.polygonData?.type === 'field') {
          return [255, 165, 0, 80]; // Orange with transparency
        } else if (d.polygonData?.type === 'zone') {
          return [0, 128, 255, 80]; // Blue with transparency
        }
        return [0, 212, 170, 80]; // Default with transparency
      }
    };
  }

  private loadData(): void {
    forkJoin({
      config: this.configService.getConfig(),
      metadata: this.metadataService.getMetadata()
    }).subscribe({
      next: ({ config, metadata }) => {
        this.config = config;
        this.tilesAvailable.set(metadata.tilesAvailable);
        this.wowTilesAvailable.set(metadata.wowTilesAvailable || false);
        this.sourceInfo.set(metadata.source);
        this.tileset = metadata.tileset;
        this.tileEndpoints = metadata.tileEndpoints;

        this.loading.set(false);

        // Add SR tiles if map is already ready
        if (this.isMapReady() && metadata.tilesAvailable) {
          this.addSRTileLayer();
        }
      },
      error: (err) => {
        console.error('Failed to load data:', err);
        this.error.set('Failed to load configuration. Is the server running?');
        this.loading.set(false);
      }
    });
  }

  /**
   * Handle fld-map ready event - following storybook pattern
   */
  onMapReady(event: boolean): void {
    this.isMapReady.set(true);
    console.log('✅ FLD Map ready');

    setTimeout(() => {
      const mapControlsService: any = _.get(this, 'mapConfiguration.services.mapControlsService');
      const deckLayersService: any = _.get(this, 'mapConfiguration.services.deckLayersService');
      const leafletService: any = _.get(this, 'mapConfiguration.services.leafletService');

      // Set up tooltip handler for polygons (from storybook)
      if (deckLayersService) {
        const deckProps: any = _.get(deckLayersService, 'deck._deck.props');
        if (deckProps) {
          deckProps.getTooltip = (data: any) => {
            if (data && data.layer && data.object) {
              const object = data.object;
              let text = '';
              if (object.polygonData) {
                text = object.polygonData.name;
              }
              if (text) {
                return {
                  html: text,
                  style: {
                    'background-color': '#1a2332',
                    color: '#f1f5f9',
                    padding: '8px 12px',
                    'border-radius': '6px',
                    'box-shadow': '0 4px 20px rgba(0, 0, 0, 0.4)',
                    'font-size': '13px',
                    'font-weight': '500',
                    border: '1px solid #2d3a4f'
                  }
                };
              }
            }
            return null;
          };
        }
      }

      // Add edit polygon controls if editing is enabled (from storybook)
      if (mapControlsService) {
        if (this.mapConfiguration.editPolygons && !_.find(mapControlsService.mapButtons, { name: 'polygonManagement' })) {
          mapControlsService.addEditPolygonsControls();
          console.log('✅ Edit polygon controls added');
        }

        // Reset button controls and update map buttons
        mapControlsService.resetButtonControls();
        mapControlsService.updateMapButtons();

        // Note: Removed addFindMyLocationControl - not needed for this POC
      }

      // Add user location marker and set up zoom tracking
      if (leafletService && leafletService.map) {
        this.addUserLocationMarker(leafletService.map);

        // Add SR tile layer if tiles are available
        if (this.tilesAvailable()) {
          this.addSRTileLayer();
        }

        // Track zoom changes
        this.currentZoom.set(Math.round(leafletService.map.getZoom()));
        leafletService.map.on('zoomend', () => {
          this.currentZoom.set(Math.round(leafletService.map.getZoom()));
        });
      }

      // Log instructions for editing
      if (this.mapConfiguration.editPolygons) {
        console.log('📝 To draw: Click Draw button (✏️) → Click on map to add vertices → Double-click to finish');
        console.log('📝 To edit: Click Edit button (🖊️) → Click polygon → Drag vertices');
      }

      console.log('✅ Map ready! Check Actions panel for events.');
    });
  }

  /**
   * Handle map change event
   */
  onMapChanged(mapName: string): void {
    console.log('Map changed to:', mapName);
  }

  /**
   * Handle find my location event
   */
  onFindMyLocation(): void {
    console.log('Find my location triggered');
    this.flyToUserLocation();
  }

  /**
   * Handle map events (polygon selection, creation, etc.) - from storybook
   */
  private handleMapEvents(event: any): void {
    if (event.eventName === 'selectPolygon') {
      console.log('✅ Polygon selected:', event.data.name, '(ID:', event.data.id + ')');
      this.updateDrawnPolygons();
    } else if (event.eventName === 'addPolygon') {
      console.log('✅ New polygon added:', event.data);
      this.updateDrawnPolygons();

      // Automatically switch to edit mode after drawing a polygon
      this.switchToEditMode();
    } else if (event.eventName === 'polygonClicked') {
      console.log('⚠️ Polygon clicked:', event.data.polygonData?.name, '- Not in edit mode');
      console.log('💡 Hint: Click Edit button (🖊️) first, then click polygon');
    } else if (event.eventName === 'editPolygon') {
      console.log('✏️ Polygon edited:', event.data);
      this.updateDrawnPolygons();
    }
  }

  /**
   * Switch to edit polygon mode programmatically
   */
  private switchToEditMode(): void {
    const deckLayersService: any = _.get(this, 'mapConfiguration.services.deckLayersService');
    const mapControlsService: any = _.get(this, 'mapConfiguration.services.mapControlsService');
    
    if (deckLayersService && mapControlsService) {
      // Switch from draw mode to edit mode
      setTimeout(() => {
        deckLayersService.editNewPolygon();
        
        // Update the UI button state to show Edit as active
        mapControlsService.polygonManagement.selectedItem = 'editPolygon';
        mapControlsService.polygonManagement.isVisible = true;
        
        // Force UI update
        if (mapControlsService.cdRef) {
          mapControlsService.cdRef.markForCheck();
        }
        
        console.log('🔄 Switched to edit mode');
      }, 100);
    }
  }

  /**
   * Update the list of drawn polygons
   */
  private updateDrawnPolygons(): void {
    const deckLayersService: any = _.get(this, 'mapConfiguration.services.deckLayersService');
    if (!deckLayersService) return;

    // Drawn polygons are stored in editPolygonData.features (GeoJSON format)
    const drawnFeatures = deckLayersService.editPolygonData?.features || [];

    // Convert GeoJSON features to polygon format for our component
    const polygons = drawnFeatures.map((feature: any, index: number) => ({
      polygon: feature.geometry?.coordinates?.[0] || [],
      polygonData: {
        id: feature.properties?.id || `drawn-${index + 1}`,
        name: feature.properties?.name || `Polygon ${index + 1}`,
        type: 'drawn',
        ...feature.properties
      }
    }));

    this.drawnPolygons.set(polygons);
    console.log(`📊 Total polygons: ${polygons.length}`, polygons);
  }

  /**
   * Add Sentinel-2 SR tile layer to the map
   */
  private addSRTileLayer(): void {
    const leafletService: any = _.get(this, 'mapConfiguration.services.leafletService');
    if (!leafletService || !leafletService.map) {
      console.warn('Leaflet service not available for SR tiles');
      return;
    }

    // Remove existing SR tile layer if any
    if (this.srTileLayer) {
      leafletService.map.removeLayer(this.srTileLayer);
      this.srTileLayer = null;
    }

    // Get tile URL from endpoints based on selected source
    let tileUrl = '/api/tiles/{z}/{x}/{y}.png';
    if (this.useWowTiles() && this.tileEndpoints?.wow) {
      tileUrl = this.tileEndpoints.wow;
    } else if (this.tileEndpoints?.original) {
      tileUrl = this.tileEndpoints.original;
    }

    // Get bounds and zoom from tileset
    const minZoom = this.tileset?.minzoom || 10;
    const maxNativeZoom = this.tileset?.maxzoom || 18;
    const bounds = this.tileset?.bounds;

    // Create tile layer options
    // maxNativeZoom = actual tile zoom level available
    // maxZoom = allows overzooming (tiles will be upscaled)
    const tileOptions: L.TileLayerOptions = {
      minZoom: minZoom,
      maxNativeZoom: maxNativeZoom,
      maxZoom: 22, // Allow overzooming up to zoom 22
      opacity: this.opacity() / 100,
      attribution: 'Sentinel-2 SR via UP42',
      tileSize: 256,
      zIndex: 100 // Above base map
    };

    // Add bounds if available
    if (bounds && bounds.length === 4) {
      tileOptions.bounds = L.latLngBounds(
        [bounds[1], bounds[0]], // southwest
        [bounds[3], bounds[2]]  // northeast
      );
    }

    // Create and add the tile layer
    this.srTileLayer = L.tileLayer(tileUrl, tileOptions);
    this.srTileLayer.addTo(leafletService.map);

    console.log('✅ SR tile layer added:', tileUrl);
    console.log('   Native zoom range:', minZoom, '-', maxNativeZoom, '(overzoom to 22)');
    if (bounds) {
      console.log('   Bounds:', bounds);
    }
  }

  /**
   * Update SR tile layer opacity
   */
  updateTileOpacity(opacity: number): void {
    this.opacity.set(opacity);
    if (this.srTileLayer) {
      this.srTileLayer.setOpacity(opacity / 100);
    }
  }

  /**
   * Handle opacity slider change
   */
  onOpacityChange(value: string | number): void {
    const opacity = typeof value === 'string' ? parseInt(value, 10) : value;
    this.updateTileOpacity(opacity);
  }

  /**
   * Set tile source (original or wow/Real-ESRGAN)
   */
  setTileSource(source: 'original' | 'wow'): void {
    const useWow = source === 'wow';
    if (this.useWowTiles() === useWow) return;

    this.useWowTiles.set(useWow);

    // Remove existing tile layer
    const leafletService: any = _.get(this, 'mapConfiguration.services.leafletService');
    if (this.srTileLayer && leafletService?.map) {
      leafletService.map.removeLayer(this.srTileLayer);
      this.srTileLayer = null;
    }

    // Re-add with new source
    this.addSRTileLayer();
    console.log('🔄 Tile source changed to:', source);
  }

  /**
   * Add user location marker as a blue pulsing point
   */
  private addUserLocationMarker(map: L.Map): void {
    // Create custom icon for user location
    const userLocationIcon = L.divIcon({
      className: 'user-location-marker',
      html: `
        <div class="user-location-pulse"></div>
        <div class="user-location-dot"></div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    });

    // Create marker
    this.userLocationMarker = L.marker([USER_LOCATION.lat, USER_LOCATION.lng], {
      icon: userLocationIcon
    }).addTo(map);

    // Add popup
    this.userLocationMarker.bindPopup(`
      <div class="user-popup">
        <strong>📍 Your Location</strong>
        <div class="coords">
          ${USER_LOCATION.lat.toFixed(6)}°N, ${Math.abs(USER_LOCATION.lng).toFixed(6)}°W
        </div>
      </div>
    `);

    console.log('✅ User location marker added at:', USER_LOCATION);
  }

  /**
   * Fly to user location
   */
  flyToUserLocation(): void {
    const leafletService: any = _.get(this, 'mapConfiguration.services.leafletService');
    if (leafletService) {
      leafletService.focusOnPoint([USER_LOCATION.lng, USER_LOCATION.lat], 15);
    }
  }

  /**
   * Zoom to data extent
   */
  zoomToExtent(): void {
    const source = this.sourceInfo();
    const leafletService: any = _.get(this, 'mapConfiguration.services.leafletService');

    if (!leafletService || !source?.bbox) return;

    const bbox = source.bbox;
    const coordinates = [
      [bbox[0], bbox[1]],
      [bbox[2], bbox[1]],
      [bbox[2], bbox[3]],
      [bbox[0], bbox[3]]
    ];

    leafletService.fitTheMap(coordinates, true, { padding: [50, 50] });
  }

  /**
   * Clear all drawn polygons
   */
  clearAllPolygons(): void {
    const deckLayersService: any = _.get(this, 'mapConfiguration.services.deckLayersService');

    if (deckLayersService) {
      // Clear the edit polygon data
      if (deckLayersService.editPolygonData) {
        deckLayersService.editPolygonData.features = [];
      }

      // Remove draw polygons layer
      deckLayersService.removeDrawPolygons();

      // Update deck layers
      if (deckLayersService.deck) {
        deckLayersService.updateLayers(deckLayersService.deck);
      }

      // Clear our local state
      this.drawnPolygons.set([]);

      console.log('🗑️ All polygons cleared');
    }
  }

  /**
   * Download drawn polygons as GeoJSON
   */
  downloadGeoJSON(): void {
    const polygons = this.drawnPolygons();
    if (polygons.length === 0) {
      console.warn('No polygons to download');
      return;
    }

    // Convert to GeoJSON FeatureCollection
    const features = polygons.map((poly: any, index: number) => {
      // Get coordinates - handle different polygon formats
      let coordinates = poly.polygon || poly.coordinates || [];

      // Ensure it's a closed ring (first point = last point)
      if (coordinates.length > 0) {
        const first = coordinates[0];
        const last = coordinates[coordinates.length - 1];
        if (first[0] !== last[0] || first[1] !== last[1]) {
          coordinates = [...coordinates, first];
        }
      }

      return {
        type: 'Feature',
        properties: {
          id: poly.polygonData?.id || `polygon-${index + 1}`,
          name: poly.polygonData?.name || `Polygon ${index + 1}`,
          type: poly.polygonData?.type || 'drawn',
          createdAt: new Date().toISOString()
        },
        geometry: {
          type: 'Polygon',
          coordinates: [coordinates]
        }
      };
    });

    const geojson = {
      type: 'FeatureCollection',
      features: features,
      metadata: {
        exportedAt: new Date().toISOString(),
        source: 'Sentinel-2 Viewer POC',
        count: features.length
      }
    };

    // Create and download file
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `polygons-${new Date().toISOString().slice(0, 10)}.geojson`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    console.log(`✅ Downloaded ${features.length} polygon(s) as GeoJSON`);
  }

  /**
   * Refresh data
   */
  refreshData(): void {
    this.loading.set(true);
    this.error.set(null);

    this.metadataService.getMetadata().subscribe({
      next: (metadata) => {
        this.loading.set(false);
        this.tilesAvailable.set(metadata.tilesAvailable);
        this.sourceInfo.set(metadata.source);
      },
      error: (err) => {
        console.error('Failed to refresh:', err);
        this.error.set('Failed to refresh data');
        this.loading.set(false);
      }
    });
  }

  ngOnDestroy(): void {
    if (this.userLocationMarker) {
      this.userLocationMarker.remove();
    }
    if (this.srTileLayer) {
      this.srTileLayer.remove();
    }
  }
}
