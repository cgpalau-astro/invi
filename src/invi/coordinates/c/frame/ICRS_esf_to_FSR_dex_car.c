//--------------------------------------------------------------
// Scale Transformations

double *ScaleT0( double *we ) { 
        we[0] = kpc2km(we[0]);
        we[1] = deg2rad(we[1]);
        we[2] = deg2rad(we[2]);
        
        we[4] = mas_yr2rad_s(we[4]);
        
        we[5] = mas_yr2rad_s(we[5]);
        
        return we;
    }

double *ScaleT1( double *we ) {
        we[0] = km2kpc(we[0]);
        we[1] = km2kpc(we[1]);
        we[2] = km2kpc(we[2]);
        
        return we;
    }

double *ScaleT2( double *we ) {
        we[3] = km_s2kpc_Myr(we[3]);
        we[4] = km_s2kpc_Myr(we[4]);
        we[5] = km_s2kpc_Myr(we[5]);
        
        return we;
    }
//--------------------------------------------------------------

//{r, delta, alpha}
double *esf2car( double *we ) {
        
        double *wb = calloc( 6 , sizeof(double) );
        
        //In: {r, t, p, pr, pt, pp}
        wb[0] = we[0]*cos(we[1])*cos(we[2]);
        wb[1] = we[0]*cos(we[1])*sin(we[2]);
        wb[2] = we[0]*sin(we[1]);
        
        wb[3] = cos(we[1])*cos(we[2])*we[3] - we[0]*sin(we[1])*cos(we[2])*we[4] - we[0]*cos(we[1])*sin(we[2])*we[5];
        wb[4] = cos(we[1])*sin(we[2])*we[3] - we[0]*sin(we[1])*sin(we[2])*we[4] + we[0]*cos(we[1])*cos(we[2])*we[5];
        wb[5] = sin(we[1])*we[3] + we[0]*cos(we[1])*we[4];
        
        int i;
        for ( i = 0; i < 6; i++ ) {
            we[i] = wb[i];
        }
        
        free(wb);
        
        return we;
    }
    
double *ICRS_esf2ICRS_car( double *we ) {
        return esf2car(we);
    }

//--------------------------------------------------------------
    
double *ICRS2GCS( double *we ) {
        
        double J[6*6] =
        {-0.05487556, -0.87343709, -0.48383502, 0.0, 0.0, 0.0,
        0.49410943, -0.44482963,  0.74698224, 0.0, 0.0, 0.0,
        -0.86766615, -0.19807637,  0.45598378, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, -0.05487556, -0.87343709, -0.48383502,
        0.0, 0.0, 0.0,  0.49410943, -0.44482963,  0.74698224,
        0.0, 0.0, 0.0, -0.86766615, -0.19807637,  0.45598378 };
        
        we = RV(we, J);
       
        return we;
    }
    
//--------------------------------------------------------------
    
double *GCS2LSR( double *we, double *prop ) {        
        
        double z_sun = prop[1];
        double mux_sun = prop[2];
        double muy_sun = prop[3];
        double muz_sun = prop[4];
        
        //Mean translation       
        //
        //
        we[2] =  we[2] + z_sun;
        we[3] =  we[3] + mux_sun;
        we[4] =  we[4] + muy_sun;
        we[5] =  we[5] + muz_sun;        
        
        //Mean reflection 
        we = Ref('x', we);
        
        return we;
    }

//--------------------------------------------------------------
    
double *LSR2FSR( double *we, double *prop ) {
        
        double x_sun = prop[0];
        double mux_o2 = 0.0;
        double muy_o2 = prop[5];
        double muz_o2 = 0.0;
        
        //Mean translation
        we[0] = we[0] + x_sun;
        //
        //
        we[3] =  we[3] + mux_o2;
        we[4] =  we[4] + muy_o2;
        we[5] =  we[5] + muz_o2;
        
        return we;
    }
    
//--------------------------------------------------------------
    
double *ICRS_esf_to_FSR_dex_car( double *we, double *prop ) {
        
        //In ICRS_esf: r [kpc], delta [deg], alpha [deg], mu^r [km/s], mu^delta [mas/yr], mu^alpha_str [mas/yr]
        we[5] = we[5]/cos(deg2rad(we[1]));
        //Out ICRS_esf: r [kpc], delta [deg], alpha [deg], mu^r [km/s], mu^delta [mas/yr], mu^alpha [mas/yr]
        
        we = ScaleT0(we);
        //Out: ICRS_esf: r [km], delta [rad], alpha [rad], mu^r [km/s], mu^delta [rad/s], mu^alpha [rad/s]
        
        //In:
        we = ICRS_esf2ICRS_car(we); 
        //Out: ICRS_car: x [km], mu [km/s]
        
        //In:
        we = ScaleT1(we);
        //Out: ICRS_car: x [kpc], mu [km/s]            
        
        //In:
        we = ICRS2GCS(we);
        //Out: GCS: x [kpc], mu [km/s]
        
        //In:
        we = GCS2LSR(we, prop);
        //Out: LSR: x [kpc], mu [km/s]
        
        //In:
        we = LSR2FSR(we, prop);
        //Out: FSR: x [kpc], mu [km/s]
        
        //In:
        we = Ref('y', we);
        //Out: FSR_dex: x [kpc], mu [km/s]
        
        //In:
        //we = ScaleT2(we);
        //Out: FSR_dex: x [kpc], mu [kpc/Myr]
        
        return we;
    }

//--------------------------------------------------------------
