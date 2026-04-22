//--------------------------------------------------------------

double kpc2km( double x ){ return x * (1000.0*30856775813057.0); }
double km2kpc( double x ){ return x / (1000.0*30856775813057.0); }

double deg2rad( double x ){ return x * (2.0*PI/360.0); }
double rad2deg( double x ){ return x / (2.0*PI/360.0); }

double mas2rad( double x ){ return x * (1.0/60.0/60.0/360.0*2.0*PI/1000.0); }
double rad2mas( double x ){ return x / (1.0/60.0/60.0/360.0*2.0*PI/1000.0); }

double inv_s2inv_yr( double x ){ return x * (60.0*60.0*365.25*24.0); }
double inv_yr2inv_s( double x ){ return x / (60.0*60.0*365.25*24.0); }

double mas_yr2rad_s( double x ){ return inv_yr2inv_s(mas2rad(x)); }
double rad_s2mas_yr( double x ){ return inv_s2inv_yr(rad2mas(x)); }

double inv_s2inv_Myr( double x ){ return x * (1.0E6*60.0*60.0*365.25*24.0); }
double inv_Myr2inv_s( double x ){ return x / (1.0E6*60.0*60.0*365.25*24.0); }

double inv_s22inv_Myr2( double x ){ return x * (1.0E6*60.0*60.0*365.25*24.0)*(1.0E6*60.0*60.0*365.25*24.0); }
double inv_Myr22inv_s2( double x ){ return x / (1.0E6*60.0*60.0*365.25*24.0)*(1.0E6*60.0*60.0*365.25*24.0); }

double km_s2kpc_Myr( double x ){ return inv_s2inv_Myr(km2kpc(x)); }
double kpc_Myr2km_s( double x ){ return inv_Myr2inv_s(kpc2km(x)); }

//--------------------------------------------------------------

double *RV( double *V, double *J ) {
        double *R = malloc( 6 * sizeof(double) );
        double r0 = 0.0, r1 = 0.0;
        int d = 3, f, c;
        
        for ( f = 0; f < d; f++ ) {
            for ( c = 0; c < d; c++ ) {
                r0 = r0 + J[6*(f) + (c)] * V[c];
                r1 = r1 + J[6*(f+d) + (c+d)] * V[c+d];
            }
            R[f] = r0;
            R[f+d] = r1;
            r0 = 0.0;
            r1 = 0.0;
        }
        
        for ( f = 0; f < 6; f++ ) {
            V[f] = R[f];
        }
        
        free(R);
        
        return V;
    }
    
//--------------------------------------------------------------

double *Ref( char axis, double *we ) {
        
        switch (axis) {
        case 'x': we[0]=-we[0]; we[3]=-we[3]; break;
        case 'y': we[1]=-we[1]; we[4]=-we[4]; break;
        }
        
        return we;
    }

//--------------------------------------------------------------

double arctan2(double y, double x) {
        double phi = atan2(y,x);
        if ( phi < 0.0 ) {
            phi = 2.0*PI + phi;
        }
        return phi;
    }
    
//--------------------------------------------------------------
