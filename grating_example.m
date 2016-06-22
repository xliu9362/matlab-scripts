clear all;


clc;

 

row=512;

col=512;

x  =0:(2*pi/(col-1)):2*pi;

y  =0:(2*pi/(row-1)):2*pi;

k=5;

coskx  = cos(k*x);

%cosky  = cos(k*y);

cosky=ones(size(x));

cosval = zeros(size(y,2),size(x,2));

 

for i=1:size(y,2)

    for j=1:size(x,2)

        cosval(i,j) = coskx(j)*cosky(i);       

    end

end


figure;
imshow(cosval,[])

%figure, plot(x,coskx,'LineWidth',3)

%hleg1 = legend('cos(x)');

%grid on;
fft_cosval=fft2(cosval);
mag_cosval=abs(fftshift(fft_cosval));
figure;
imshow(mag_cosval);