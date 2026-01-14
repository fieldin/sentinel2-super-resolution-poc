import { Routes } from '@angular/router';
import { MapComponent } from './map/map.component';
import { EnhanceComponent } from './enhance/enhance.component';

export const routes: Routes = [
  {
    path: '',
    component: MapComponent,
    title: 'Sentinel-2 Viewer'
  },
  {
    path: 'enhance',
    component: EnhanceComponent,
    title: 'Real-ESRGAN Enhancer'
  },
  {
    path: '**',
    redirectTo: ''
  }
];

