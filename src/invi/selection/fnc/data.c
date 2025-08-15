
double deg2rad(double x){ return x * (2.0*PI/360.0); }
double mas2rad(double x){ return x * (1.0/60.0/60.0/360.0*2.0*PI/1000.0); }

//----------------------------------------------------------------------------

#define w(f) (ws.w[f])
#define S(f,c) (ws.S[D*(f) + (c)])

struct WS symmetrize(struct WS ws) {

    //Copy superior to inferior triangular
    for (int f = 0; f < D; f++) {
        for (int c = f+1; c < D; c++) {
            S(c,f) = S(f,c);
        }
    }

    return ws;
}


struct WS split_line_ob(char *line, struct WS ws) {

    //Read diagonal and superior triangular
    sscanf(line, "%lf%lf%lf%lf%lf%lf %lf%lf%lf%lf%lf%lf %lf%lf%lf%lf%lf %lf%lf%lf%lf %lf%lf%lf %lf%lf %lf", &w(0), &w(1), &w(2), &w(3), &w(4), &w(5),     &S(0,0), &S(0,1), &S(0,2), &S(0,3), &S(0,4), &S(0,5),     &S(1,1), &S(1,2), &S(1,3), &S(1,4), &S(1,5),      &S(2,2), &S(2,3), &S(2,4), &S(2,5),     &S(3,3), &S(3,4), &S(3,5),      &S(4,4), &S(4,5),       &S(5,5) );

    //Copy superior to inferior triangular
    ws = symmetrize(ws);

    return ws;
}


struct WS split_line_data(char *line, struct WS ws, int include_rv) {

    //Read diagonal and superior triangular
    sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &w(0), &w(1), &w(2), &w(3), &w(4), &w(5), &S(0,0), &S(0,1), &S(0,2), &S(0,4), &S(0,5), &S(1,1), &S(1,2), &S(1,4), &S(1,5), &S(2,2), &S(2,4), &S(2,5), &S(3,3), &S(4,4), &S(4,5), &S(5,5));

    //Include radial velocity
    if ( include_rv == 1 ) {
        if ( isnan(w(3)) == 1 ) {
            w(3) = 0.0;
            S(3,3) = 1000.0;
        }
    }

    //Do not include radial velocity
    if ( include_rv == 0 ) {
        w(3) = 0.0;
        S(3,3) = 1000.0;
    }

    //Units
    w(1) = deg2rad(w(1));
    w(2) = deg2rad(w(2));

    S(1,1) = mas2rad(S(1,1));
    S(2,2) = mas2rad(S(2,2));

    //Star
    double cosd = cos(w(1));
    w(5) = w(5)/cosd;
    S(2,2) = S(2,2)/cosd;
    S(5,5) = S(5,5)/cosd;

    //Real Values
    S(0,1) = S(0,1)*S(0,0)*S(1,1);
    S(0,2) = S(0,2)*S(0,0)*S(2,2);
    S(0,4) = S(0,4)*S(0,0)*S(4,4);
    S(0,5) = S(0,5)*S(0,0)*S(5,5);

    S(1,2) = S(1,2)*S(1,1)*S(2,2);
    S(1,4) = S(1,4)*S(1,1)*S(4,4);
    S(1,5) = S(1,5)*S(1,1)*S(5,5);

    S(2,4) = S(2,4)*S(2,2)*S(4,4);
    S(2,5) = S(2,5)*S(2,2)*S(5,5);

    S(4,5) = S(4,5)*S(4,4)*S(5,5);

    //Square diagonal elements
    for (int i = 0; i < D; i++) {
        S(i,i) = S(i,i)*S(i,i);
    }

    //Copy superior to inferior triangular
    ws = symmetrize(ws);

    //Table Conversion
    //In ICRS_esf: pi [mas], delta [deg], alpha [deg], mu^r [km/s], mu^delta [mas/yr], mu^alpha* [mas/yr]
    //In ICRS_esf: pi_e [mas], delta_e [mas], alpha_e* [mas], mu^r_e [km/s], mu^delta_e [mas/yr], mu^alpha*_e [mas/yr] //Errors are not squared
    //ws = Table_Conversion_Errors(ws);
    //Out ICRS_esf: pi [mas], delta [rad], alpha [rad], mu^r [km/s], mu^delta [mas/yr], mu^alpha [mas/yr]
    //Out ICRS_esf: pi_e [mas], delta_e [rad], alpha_e [rad], mu^r_e [km/s], mu^delta_e [mas/yr], mu^alpha_e [mas/yr] //Errors are not squared

    //Square sigmas
    //ws = square_s(ws);

    return ws;
}

#undef w
#undef S

//----------------------------------------------------------------------------
//Load volume defined by orbit bundle

struct WS *load_volume_ob(char *file_ob, int n) {

    char line[1024];

    struct WS *volume_ob = Init_WS(n);

    FILE *f_ob = fopen(file_ob, "r");
    for (int i = 0; i < n; i++) {
        fgets(line, sizeof(line), f_ob);
        volume_ob[i] = split_line_ob(line, volume_ob[i]);
    }
    fclose(f_ob);

    return volume_ob;
}

//----------------------------------------------------------------------------
