%==========================================================================
% this function returns an array of the temporal CF
%==========================================================================
function CF= temporal_correlation_fn(s1,s2,m1,m2)
delta_t=0.2e-3; % time resolution in seconds;
fs=1/delta_t; %sampling frequency
%w_max=pi*fs;
w_max=fs*pi;
N=14001*5; % number of samples;
delta_w=(2*pi)/(N*delta_t); % frequency resolution
% frequency range
w_range=(0:delta_w:(N/2)*delta_w);
t_range=(0:delta_t:(N-1)*delta_t);
a=2e-3; %m;
K_lowest=(2*pi)/a; %m^-1 
% various K terms . 
%k=[0 0;K_lowest 0; 0 K_lowest;];
%k=[0 0;K_lowest 0; 2*K_lowest 0; 3*K_lowest 0;0 K_lowest;0 2*K_lowest;K_lowest K_lowest;2*K_lowest K_lowest;K_lowest 2*K_lowest;K_lowest 3*K_lowest;];
%k=[0 0;K_lowest 0; 2*K_lowest 0;0 K_lowest;2*K_lowest 0;K_lowest K_lowest;2*K_lowest K_lowest;K_lowest 2*K_lowest;2*K_lowest 2*K_lowest;K_lowest 3*K_lowest;2*K_lowest 3*K_lowest;3*K_lowest 3*K_lowest;];
% this combination of K terms are used to plot the temporal CF plot
%k=[0 0;K_lowest 0;2*K_lowest 0; 2*K_lowest 0; 3*K_lowest 0;0 K_lowest;0 2*K_lowest;0 3*K_lowest; K_lowest K_lowest;2*K_lowest K_lowest;3*K_lowest K_lowest;K_lowest 2*K_lowest;2*K_lowest 2*K_lowest;3*K_lowest 2*K_lowest;K_lowest 3*K_lowest;2*K_lowest 3*K_lowest;3*K_lowest 3*K_lowest;K_lowest 4*K_lowest; 2*K_lowest 4*K_lowest;3*K_lowest 4*K_lowest;K_lowest 5*K_lowest; 2*K_lowest 5*K_lowest;3*K_lowest 5*K_lowest;];
% this combination of K is used  to produce the CF_contour plot of varying M2
k=[0 0;K_lowest 0; 0 K_lowest;K_lowest K_lowest;K_lowest 2*K_lowest;];  
r=m1;
r_=m2;
r2=s2;
r1=s1;

R1=r-r1; %m m1-s1 
R2=r_-r1;%m m2-s1
R3=r-r2; %m m1-s2
R4=r_-r2; %m m2-s2

%T_es(m1-s1,w)
T_R1=TransferFn(k,R1,w_range);
%T_es(m2-s1,w)
T_R2=TransferFn(k,R2,w_range);
%T_es(m1-s2,w)
T_R3=TransferFn(k,R3,w_range);
%T_es(m2-s2,w)
T_R4=TransferFn(k,R4,w_range);

%==========================================================================
%{
% This part of code is for debugging the Transfer functiion 
figure; 
w_plot_half=linspace(0,w_max/(2*pi),(N+1)/2);
subplot(3,2,1);plot(w_plot_half,T_R1);title('T_{es}(m1-s1, \omega)')
subplot(3,2,2);plot(w_plot_half,T_R2);title('T_{es}(m1-s2, \omega)');
subplot(3,2,3);plot(w_plot_half,T_R3);title('T_{es}(m2-s1, \omega)');
subplot(3,2,4);plot(w_plot_half,T_R4);title('T_{es}(m2-s2, \omega)');
subplot(3,2,5);plot(w_plot_half,T_R1.*T_R2);title('T_{es}(m1-s1, \omega).*T_{es}(m1-s2, \omega)');
subplot(3,2,6);plot(w_plot_half,T_R3.*T_R4);title('T_{es}(m2-s1, \omega).*T_{es}(m2-s2, \omega)');
%}
%===========================================================================
CorrFn_freq=T_R1.*conj(T_R2)+T_R3.*conj(T_R4);
CorrFn_freq=[fliplr(CorrFn_freq(2:length(CorrFn_freq))) CorrFn_freq(1:length(CorrFn_freq))];
CorrFn_time=ifft(ifftshift(CorrFn_freq));
CF=CorrFn_time;
end