#include <stdio.h>
#include <math.h>



// Array access macros
#define INPUT(i,j) A[(i) + (m)*(j)]
#define OUTPUT(i,j) B[(i) + (m)*(j)]
#define H(i,j) Gauss[(i) + (Reg_x)*(j)]



/*----------------------------------------- 9 Oρίσματα στον kernel----------------------------------------  */
__global__ void myKernel(float const* const A, float *B, float const* const Gauss, int const m, int const n, int const thrds, float const s, int const Reg_x, int const Reg_y)
{

/* Συντεταγμένες του κάθε νήματος και του κάθε επιμέρους block στο grid. */
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x;
      int l = blockIdx.y;

	int a,b,e,g,c,d;
	int xx, yy, xk, yk, xxk, yyk;	
	float w,w1,z1;
	float norm=0;
	float Z_total = 0;
	float subs;
	
	if(k<1 && l<1)
	{
		w1 = m/thrds ;
		z1 = n/thrds ;
		

		for(e=0 ; e<w1 ; e++)
		{
			for(g=0 ; g<z1 ; g++)
			{	
				Z_total = 0 ;
				xx = threadIdx.x*w1 + e;		// Συντεταγμένες του κέντρου καθενός στοιχείου.
				yy = threadIdx.y*z1 + g;
				for(a=0; a<m ; a++)
				{
					for(b=0; b<n ; b++)
					{	
						norm = 0 ;
						for(c=0 ; c<Reg_x ; c++)
						{
							for(d=0 ; d<Reg_y ; d++)
							{
					
								xk = xx - Reg_x/2 + c ;		// Γείτονες του πίξελ που εξετάζεται τώρα.
								yk = yy - Reg_y/2 + d ;
					
								xxk = a - Reg_x/2 + c ;
								yyk = b - Reg_y/2 + d ;	// Συντεταγμένες των γειτονικών του καθενός pixel.
					 /* Έλεγχος συνθηκών εάν βγούμε εκτός πίνακα.Τότε χρησιμοποιούμε κατοπτρικές συνθήκες. */
								if(xk<0)
									xk = fabsf(xk);
								if(xk>m)
									xk = m-(xk-m);
								if(yk<0)
									yk = fabsf(yk);
								if(yk>n)
									yk = n-(yk-n);
					
					/* Έλεγχος των συνθηκών κάποιο γειτονικό pixel του τρέχοντος προς εξέταση να βγαίνει εκτός του πίνακα. */
								if(xxk<0)
									xxk = fabsf(xxk);
								if(xk>m)
									xxk = m-(xxk-m);
								if(yyk<0)
									yyk = fabsf(yyk);
								if(yk>n)
									yyk = n-(yyk-n);
						
								subs = INPUT(xk,yk) - INPUT(xxk,yyk);	/* Αφαίρεση στοιχείο προς στοιχείο τα γειτονικά. */
							/* τώρα το πολλαπλασιάζουμε στοιχείο προς στοιχείο με τον fspecial,δηλ τη Gaussian. */
							// Για μεγαλύτερη ταχύτητα προτιμήθηκε να γίνει ο πολλαπλασιασμός κατευθείαν.
					
								subs = H(c,d)*subs ;
								subs = powf(subs,2);  // Έγινε και το τετράγωνο του στοιχείου.Στο τέλος θα προστεθούν όλα για να προκύχψει η νόρμα.
								norm += subs ;		  // Είναι η νόρμα του ενός πίξελ με ένα μόνο πίξελ από όλα.ΠΡΟΣΟΧΗ!

							}
						}
						Z_total += expf(-norm/s);												
						w = expf(-norm/s) ;
						OUTPUT(xx,yy) += w*INPUT(a,b) ;
					}

				}
				
				OUTPUT(xx,yy) = (OUTPUT(xx,yy)/Z_total) ;
				__syncthreads() ;
			}
		}






	}

}
