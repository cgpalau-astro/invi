//-------------------------------------------------------------- 
//ICRS_car2ICRS_esf

//{r, delta, alpha}
double *car2esf_Inv( double *we ) {
        
        we[0] = kpc2km(we[0]);
        we[1] = kpc2km(we[1]);
        we[2] = kpc2km(we[2]);
        
        double *wb = calloc( 6 , sizeof(double) );
        
        //In: {x, y, z, px, py, pz}
        wb[0] = sqrt(we[0]*we[0] + we[1]*we[1] + we[2]*we[2]);
        wb[1] = asin(we[2]/wb[0]);
        wb[2] = arctan2(we[1],we[0]);
        
        wb[3] = cos(wb[2])*cos(wb[1])*we[3] + sin(wb[2])*cos(wb[1])*we[4] + sin(wb[1])*we[5];
        wb[4] = -cos(wb[2])*sin(wb[1])/wb[0]*we[3] - sin(wb[2])*sin(wb[1])/wb[0]*we[4] + cos(wb[1])/wb[0]*we[5];
        wb[5] = -sin(wb[2])/(wb[0]*cos(wb[1]))*we[3] + cos(wb[2])/(wb[0]*cos(wb[1]))*we[4];
        
        
        wb[0] = km2kpc(wb[0]);
        
        int i;
        for ( i = 0; i < 6; i++ ) {
            we[i] = wb[i];
        }
        
        free(wb);
        
        return we;
    }

double *ICRS_car2ICRS_esf( double *we ) {
        return car2esf_Inv(we);
    }

//-------------------------------------------------------------- 
//GCS2ICRS

double *GCS2ICRS( double *we ) {
        
        double J[6*6] =
        {-0.05487556043, 0.494109428, -0.8676661489, 0., 0., 
        0.,-0.8734370902, -0.4448296299, -0.1980763735, 0., 0., 0., -0.4838350155, 
        0.7469822444, 0.4559837763, 0., 0., 0.,0., 0., 0., -0.05487556043, 
        0.494109428, -0.8676661489, 0., 0., 
        0., -0.8734370902, -0.4448296299, -0.1980763735, 0., 0., 0., -0.4838350155, 
        0.7469822444, 0.4559837763};
        
        we = RV(we, J);
        
        return we;
    }

//--------------------------------------------------------------
//LSR2GCS

double *LSR2GCS( double *we, double *prop ) {        
        double z_sun = prop[1];
        double mux_sun = prop[2];
        double muy_sun = prop[3];
        double muz_sun = prop[4];
        
        //Mean reflection 
        we = Ref('x', we);
        
        //Mean translation       
        //
        //
        we[2] =  we[2] - z_sun;
        we[3] =  we[3] - mux_sun;
        we[4] =  we[4] - muy_sun;
        we[5] =  we[5] - muz_sun;
        
        return we;
    }

//-------------------------------------------------------------- 
//FSR2LSR

double *FSR2LSR( double *we, double *prop ) {
        
        double x_sun = prop[0];
        double mux_o2 = 0.0;
        double muy_o2 = prop[5];
        double muz_o2 = 0.0;
        
        //Mean translation
        we[0] = we[0] - x_sun;
        //
        //
        we[3] =  we[3] - mux_o2;
        we[4] =  we[4] - muy_o2;
        we[5] =  we[5] - muz_o2;
        
        return we;
    }

//--------------------------------------------------------------
// Scale Transformations

double *ScaleT0_Inv( double *we ) { 
        //we[0] = km2kpc(we[0]);
        we[1] = rad2deg(we[1]);
        we[2] = rad2deg(we[2]);
        
        we[4] = rad_s2mas_yr(we[4]);        
        we[5] = rad_s2mas_yr(we[5]);
        
        return we;
    }
   
double *ScaleT2_Inv( double *we ) {
        we[3] = kpc_Myr2km_s(we[3]);
        we[4] = kpc_Myr2km_s(we[4]);
        we[5] = kpc_Myr2km_s(we[5]);        
        return we;
    }

//--------------------------------------------------------------
    
double *FSR_dex_car_to_ICRS_esf( double *we, double *prop ) {
        
        //In: FSR_dex: x [kpc], mu [kpc/Myr]
        //we = ScaleT2_Inv(we);
        //Out: FSR_dex: x [kpc], mu [km/s]
        
        we = Ref('y', we);
        //Out: FSR: x [kpc], mu [km/s]
        
        we = FSR2LSR(we, prop);
        //Out: LSR: x [kpc], mu [km/s]
        
        we = LSR2GCS(we, prop);
        //Out: GCS: x [kpc], mu [km/s]
        
        we = GCS2ICRS(we);
        //Out: ICRS: x [kpc], mu [km/s]
        
        we = ICRS_car2ICRS_esf(we);
        //Out: ICRS_esf: r [kpc], delta [rad], alpha [rad], mu^r [km/s], mu^delta [rad/s], mu^alpha [rad/s]
        
        we = ScaleT0_Inv(we);
        //Out ICRS_esf: r [kpc], delta [deg], alpha [deg], mu^r [km/s], mu^delta [mas/yr], mu^alpha [mas/yr]
        
        we[5] = we[5]*cos(deg2rad(we[1]));
        //Out ICRS_esf: r [kpc], delta [deg], alpha [deg], mu^r [km/s], mu^delta [mas/yr], mu^alpha_str [mas/yr]
        
        return we;
    }

//--------------------------------------------------------------
