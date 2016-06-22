function my_atan=myatan(y,x)
if nargin==1 %just in case the user only gives the value of y myatan(y)
    x=0; % if only y value is given, assume it's in Y axis.
end
my_atan=nan;

if(x==0 && y==0)
 % set output as NaN if input is the origin(0,0)
else
    my_atan=atan2(y,x);
end
if my_atan<0
    my_atan=my_atan+2*pi;
end
%alternative solution without using atan2
%{ 
if x>0
    my_atan=atan(y/x);
elseif y>=0 && x<0
    my_atan=pi+atan(y/x);

elseif y<0 && x<0
    my_atan=-pi+atan(y/x);

elseif y>0 && x==0
    my_atan=pi/2;

elseif y<0 && x==0
    my_atan=-pi/2;
end
if my_atan<0
    my_atan=my_atan+2*pi;
end
%}


end
