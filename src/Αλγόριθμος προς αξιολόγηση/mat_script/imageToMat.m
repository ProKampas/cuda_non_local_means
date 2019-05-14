k = double(imread('Poke Ball.png'));
k = k./256;

imageLength = 128;
for i=1:imageLength
    for j=1:imageLength
        PokeBall(i,j) = k(i,j);
    end
end

save PokeBall.mat PokeBall

clear all