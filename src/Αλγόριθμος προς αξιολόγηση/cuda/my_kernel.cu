#include <math.h>
#include <stdio.h>

// Array access macros
#define INPUT(i,j) A[(i) + (m + 2*(Reg_x/2))*(j)]
#define OUTPUT(i,j) B[(i) + (m)*(j)]
#define H(i,j) Gauss[(i)*(Reg_y) + (j)]
#define thrds 16
#define m 128
#define n 128
#define blocks 16
__global__ void myKernel(float const* const A, float *B, float const* const Gauss, int const Reg_x, int const Reg_y, float const s) {
  // Get pixel (x,y) in input
  
  
  
  /* Κάθε block αποτελείται από επιμέρους νήματα, όσα ορίζονται κάθε φορά.
     Αντιστοιχίζουμε κάθε block με ένα pixel ώστε για κάθε ένα από αυτά να 
     γίνονται οι απαραίτητοι υπολογισμοί. Μέσα σε κάθε νήμα γίνονται υπολογισμοί
     για έναν συγκεκριμένο αριθμό pixels.	 */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x;
  int l = blockIdx.y;
  int x,y,xk,yk,xx,yy,xxk,yyk;
  int w1,w2,w3,e,g,c,d;
  /* Πίνακες οι οποίοι θα είναι ορατοί από τα νήματα του κάθε block 
     ορίζονται ως __shared__ . */
	// __shared__ float distances[m*n];
	 __shared__ float output[thrds][thrds];
	 __shared__ float z_per_block[thrds][thrds];
	 __shared__ float Z_total[m/blocks][n/blocks];		//Για κάθε pixel μέσα σε ένα block.
//	 __shared__ float output_total[m/thrds][n/thrds];
	 
		 
	 float w;
	 float z_per_thread=0.0;
	 float subs;
	 float norm=0.0;	//local variable for every thread.
  if( k<m && l<n) {
	/* Θα χρησιμοποιηθεί μια διπλή επανάληψη για κάθε νήμα ώστε
       για κάθε γειτονιά καθενός pixel να υπολογιστεί ο μέσος όρος.
	   Έτσι θα μπορώ να τον χρησιμοποιήσω αργότερα χωρίς προβλήματα.
	   Ίσως χρειαστεί κι συγχρονισμός των νημάτων. */
	   
	  /* Έχουν ήδη οριστεί από τον κώδικα στο Matlab οι διαστάσεις του κάθε block 
	  και συνεπώς πόσα pixel θα επεξεργαστούν από κάθε νήμα. */
		
	 /* Για αρχή σε κάθε block υπολογίζουμε την απόσταση του pixel που θέχουμε ως κεντρικό με όλα τα υπόλοιπα. */
	
	w2 = m/gridDim.x ;	// Για 16 νήματα και 64 στοιχεία έχουμε σε κάθε νήμα 4 pixel κλπ. , 8 , 16
	w1 = w2*m/thrds ;
	w3 = thrds/w2 ;	//Για 64x64 --> 4  ,  για 128x128 --> 2    και για 256x256 --> 1
		
	z_per_thread=0.0;
     /* Μηδενισμός του πίνακα Ζ_total στον οποίο περνούν τα συνολικά Ζ για κάθε πίξελ
		  που υπολογίζουμε μέσα σε ένα μπλοκ. */

	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		for( e=0 ; e<w2 ; e++)
		{
			for( g=0 ; g<w2 ; g++)
			{
				Z_total[e][g] = 0 ;
//				output_total[g][e] = 0;
			}
		}


		for(e=0; e<thrds; e++)
		{
			for(g=0 ; g<thrds ; g++)
			{
				output[e][g] = 0;
			}		
		}	
	}
		  
	__syncthreads();	// Απαραίτητος ο συγχρονισμός των νημάτων ώστε να έχουν σίγουρα μηδενιστεί οι παραπάνω πίνακες
	
	//Kέντρο του κάθε στοιχείου προς εξέταση. 
	x = blockIdx.x*(m/blocks) + threadIdx.x/w3 + Reg_x/2 ;
	y = blockIdx.y*(n/blocks) + threadIdx.y/w3 + Reg_y/2 ;
	/* Σε κάθε νήμα θα υπολογιστεί :
							   το 1/16 του pixel για  64χ64 ,δηλαδή ανά m/4,n/4
							   το 1/4 του pixel για 128χ128 ,δηλαδή ανά m/2,n/2
							   το 1/1 του pixel για 256χ256 ,δηλαδή ανά m/1,n/1 	*/
	for( e=0 ; e<(m/w3) ; e++)
	{
		for( g=0 ; g<(n/w3) ; g++)
		{	
			norm = 0.0 ;
			xx = (threadIdx.x % w3)*w1 + e + Reg_x/2 ;		// Συντεταγμένες του κέντρου καθενός στοιχείου.
			yy = (threadIdx.y % w3)*w1 + g + Reg_y/2 ;
			for( c=0 ; c<Reg_x ; c++)
			{
				for( d=0 ; d<Reg_y ; d++)
				{
					
					xk = x - Reg_x/2 + c ;		// Συντεταγμένες των γειτονικών του κέντρου του pixel x,y.
					yk = y - Reg_y/2 + d ;
					
					xxk = xx  - Reg_x/2 + c ;
					yyk = yy  - Reg_y/2 + d ;	// Συντεταγμένες των γειτονικών του καθενός pixel.
			 					
					subs = INPUT(xk,yk) - INPUT(xxk,yyk);	/* Αφαίρεση στοιχείο προς στοιχείο τα γειτονικά. */
					/* τώρα το πολλαπλασιάζουμε στοιχείο προς στοιχείο με τον fspecial,δηλ τη Gaussian. */
					// Για μεγαλύτερη ταχύτητα προτιμήθηκε να γίνει ο πολλαπλασιασμός κατευθείαν.
					
					subs = H(c,d)*subs ;
					subs = powf(subs,2);  // Έγινε και το τετράγωνο του στοιχείου.Στο τέλος θα προστεθούν όλα για να προκύχψει η νόρμα.
					norm += subs ;

				}
			}
			/* Πλέον έχουμε τη νόρμα μεταξύ των 2 pixel.
		   και θα περάσουμε το στοιχείο αυτό σε έναν πίνακα mxn ο οποίος στο αντίστοιχο σημείο.
		   Ωστόσο, ο πίνακας αυτός θα είναι συμμετρικός με τα διαγώνια στοιχεία΄ του μηδενικά.
		   Ένα σημείο το οποίο χρειάζεται βελτίωση.*/

			w = expf(-norm/s);
			z_per_thread += expf(-norm/s) ;
			output[threadIdx.x][threadIdx.y] += w*INPUT(xx,yy);

			//Ο παραπάνω είναι ένας πίνακας για κάθε block.
			
			//Με αυτόν τον τρόπο βρήκαμε τις αποστάσεις μεταξύ των pixels.
			//Επόμενο στοιχείο που πρέπει να βρεθεί είναι ο Ζ πίνακας στη συνέχεια ο W και τέλος ο f^.
			
			
		}
	}


	z_per_block[threadIdx.x][threadIdx.y] = z_per_thread;
	__syncthreads() ;
	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		for( e=0 ; e<thrds ; e++)
		{
			for( g=0 ; g<thrds ; g++)
			{
				Z_total[e/w3][g/w3] += z_per_block[e][g] ;
			//	output_total[e/w2][g/z2] += output[e][g] ;
			}
		}	
	}
	
	
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		for( e=0 ; e<thrds ; e++)
		{
			for( g=0 ; g<thrds ; g++)
			{
				OUTPUT(blockIdx.x*(m/blocks)+(e/w3) , blockIdx.y*(n/blocks)+(g/w3)) += (1/Z_total[e/w3][g/w3])*output[e][g] ;	//Εάν loops 0-15
		//		OUTPUT(x+e , y+g) = (1/Z_total[e][g])*output_total[e][g];	// Εάν loops 0-3
			}
		}
	}




  }
}
