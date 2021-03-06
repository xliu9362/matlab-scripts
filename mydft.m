function my_dft=mydft(a)
%a2 = 2D-DFT of  given 2D matrix  'a'
m=size(a,1);
n=size(a,2);
a1=zeros(m,n);
a2=zeros(m,n);
%fourier transform along the rows 
for i=1:n
a1(:,i)=exp(2*(-1j)*pi*(0:m-1)'*(0:m-1)/m)*a(:,i) ;
end
% fourier transform along the columns
for j=1:m
a2temp=exp(2*(-1j)*pi*(0:n-1)'*(0:n-1)/n)*(a1(j,:)).' ;
a2(j,:)=a2temp.';
end
for i = 1:m
    for j = 1:n
  if((abs(real(a2(i,j)))<0.0001)&&(abs(imag(a2(i,j)))<0.0001))
       a2(i,j)=0;
   elseif(abs(real(a2(i,j)))<0.0001)
     a2(i,j)= 0+(-1j)*imag(a2(i,j));
    elseif(abs(imag(a2(i,j)))<0.0001)
             a2(i,j)= real(a2(i,j))+0;
      end
   end
end
my_dft=a2;

