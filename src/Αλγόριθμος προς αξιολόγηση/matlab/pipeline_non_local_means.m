  %%------------------------------------------------------------
%	ARISTOTLE UNIVERSITY OF THESSALONIKI
%	       FACULTY OF ENGINEERING
%	 ELECTRICAL AND COMPUTER ENGINEERING
%
%
% AUTHOR	 Kampas Prodromos				pskampas@auth.gr
%	
% Date:	January 29,2017
% ------------------------------------------------------------
%
  
  clear all %#ok
  close all

  %% PARAMETERS
  
  % input image
  pathImg   = '../data/PokeBall.mat';
  strImgVar = 'PokeBall';
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
				
  % filter sigma value
  filtSigma = 0.02;
  patchSize = [7 7];
  patchSigma = 5/3;
  
  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));
  
  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% INPUT DATA
  
  fprintf('...loading input data...\n')
  
  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar);
  
  %% PREPROCESS
  
  fprintf(' - normalizing image...\n')
  I = normImg( I );
  
  figure('Name','Original Image');
  imagesc(I); axis image;
  colormap gray;
  saveas(gcf,'Original.png');
  %% NOISE
  
  fprintf(' - applying noise...\n')
  J = imnoise( I, noiseParams{:} );
  figure('Name','Noisy-Input Image');
  imagesc(J); axis image;
  colormap gray;
  saveas(gcf,'noise.png');
  
  %% NON LOCAL MEANS
%  fprintf(' - nonlocalmeans MATLAB time...\n')
%  tic;
%  If = nonLocalMeans( J, patchSize, filtSigma, patchSigma );
%  toc

  %% PARAMETERS
  Reg_x = (patchSize(1)) ;
  Reg_y = (patchSize(2)) ;	
  m=128;
  n=128;
  D = floor(Reg_x/2);
  
  % Mirror padding του πινακα J ώστε να ελαχιστοποιηθεί ο χρόνος επεξεργασίας από την κάρτα γραφικών.
  for i=1:(m+2*(D))
  	for j=1:(n+2*(D))
		if(i<(1+D) && j<(1+D))
			JN(i,j) = J(i+(D) , j+(D));
		
		elseif(i>(m+D) && j>(n+D))
			JN(i,j) = J(i-2*(D) , j-2*(D)) ;
		
		
		elseif(i<(1+D) && j>(n+D))
			JN(i,j) = J(i+(D) , j-2*(D));
		
		elseif(i>(m+D) && j<(1+D))
			JN(i,j) = J(i-2*(D) , j+(D));
		
		
		elseif(i<(1+D) &&  j<=(m+D) && j>=(1+D))
			JN(i,j) = J(i+(D) , j-D);
		

		elseif(i>(m+D) &&  j<=(m+D) && j>=(1+D))
			JN(i,j) = J(i-2*(D) , j-D) ;
		
		
		elseif(j>(n+D) &&  i<=(m+D) && i>=(1+D))
			JN(i,j) = J(i-D , j-2*(D));
		
		elseif(j<(1+D) &&  i<=(m+D) && i>=(1+D))
			JN(i,j) = J(i-D , j+(D));
		
		elseif(i>=(1+D) && i<=(m+D) && j>=(1+D) && j<=(m+D))
			JN(i,j) = J(i-D , j-D); 
	     end
	end	
  end
  




  % Έλεγχος ότι τα στοιχεία του πίνακα J αντιγράφηκαν στις κατάλληλες θέσεις στον πίνακα JN
  for i=1:m
	for j=1:n
		JJ(i,j) = JN(i+D,j+D);
	end
  end

  MAXX = max(JJ-J)
  
  
  %% Threads per Block
  threadsPerBlock = [16 16];

  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% KERNEL
  
  k = parallel.gpu.CUDAKernel( '../cuda/my_kernel.ptx', ...
                               '../cuda/my_kernel.cu');
  
  fprintf('...after cuda kernel %s...\n',mfilename);  
  numberOfBlocks  = [16 16];
  
  k.ThreadBlockSize = threadsPerBlock;
  k.GridSize        = numberOfBlocks;
  
  %% DATA
  regionx = patchSize(1);
  regiony = patchSize(2);
  
  % gaussian patch
  H = fspecial('gaussian',patchSize, patchSigma);
  H = H(:) ./ max(H(:));
  
  A = single(JN);
  Zero = zeros(m,n);
  B = Zero;
  B = single(Zero);
  H = single(H);
  H = H';
  fprintf('...after setting arrays %s...\n',mfilename);  
  
  %Εισαγωγή της θορυβοποιημένης εικόνας στην GPU
  tic
  A = gpuArray(A);
  B = gpuArray(B);
  H = gpuArray(H);
  wait(gpuDevice);
  toc

  fprintf('...finish transfer of arrays to GPU %s...\n',mfilename);

  
  tic
  B =  feval(k, A, B, H, regionx, regiony, filtSigma) ;
  wait(gpuDevice);
  toc
  tic
  X = gather(B);
  toc
  %% VISUALIZE RESULT
  
  figure('Name', 'Filtered image');
  imagesc(X); axis image; 
  colormap gray;
  saveas(gcf,'Filtered.png');
  
  figure('Name', 'Residual');
  imagesc(X-J); axis image;
  colormap gray;
  saveas(gcf,'Residual.png');
  
  fprintf('...the biggest error is %s...\n',mfilename);
  Error = max(max(X-J)) 

  fprintf('...the mean value is %s...\n',mfilename);
  Mean = mean(mean(X-J)) 
  %% (END)
  

  fprintf('...end %s...\n',mfilename);
  
  
%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 28, 2016
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------
