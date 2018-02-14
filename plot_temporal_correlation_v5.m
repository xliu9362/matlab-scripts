%=============================================
% OP=90, 
% s1&S2 in parallel
% short range separation (m1-s1)=(m2-s2)~2mm
%===============================================
m1_s=[1.2e-3 1.5e-3];%m measurement point #1
m2_s=[2.7e-3 1.5e-3];%m measurement point #2
s2_s=[2.7e-3 3.5e-3];%m source piont #2
s1_s=[1.2e-3 3.5e-3];%m  source point #1
CF1=temporal_correlation_fn(s1_s,s2_s,m1_s,m2_s);

%=============================================
% OP=0, 
% s1&S2 inline
% short range separation (m1-s1)=(m2-s2)~2mm
%===============================================

% theta=0,(m1-s1)=(m2-s2)~2mm;
m1_0=[0.2e-3 1.3e-3];%m measurement point #1
m2_0=[4.3e-3 1.3e-3];%m measurement point #2
s2_0=[2.3e-3 1.5e-3];%m source piont #2
s1_0=[1.7e-3 1.5e-3];%m  source point #1
CF2=temporal_correlation_fn(s1_0,s2_0,m1_0,m2_0);
%=============================================
% OP=45, 
% s1&S2 in parallel
% short range separation (m1-s1)=(m2-s2)~2mm
%===============================================
% theta=45,(m1-s1)=(m2-s2)~2mm;
m1_45=[4.4e-3 3.9e-3];%m measurement point #1
m2_45=[3.4e-3 3.9e-3];%m measurement point #2
s2_45=[0.5e-3 2.1e-3];%m source piont #2
s1_45=[-0.5e-3 2.1e-3];%m  source point #1
CF3=temporal_correlation_fn(s1_45,s2_45,m1_45,m2_45);
%=============================================
% OP_s1=90, OP_s2=0; 
% s1&S2 inline
% short range separation (m1-s1)=(m2-s2)~2mm
%===============================================
%(m1-s1)=(m2-s2)~2mm;
m1_90=[1.2e-3 1.5e-3];%m measurement point #1
m2_0=[4.3e-3 3.5e-3];%m measurement point #2
s2_0=[2.3e-3 3.5e-3];%m source piont #2 OP_s2=0
s1_90=[1.2e-3 3.5e-3];%m  source point #1 OP_s1=90
CF4=temporal_correlation_fn(s1_90,s2_0,m1_90,m2_0);


delta_t=0.2e-3; % time resolution in seconds;
fs=1/delta_t; %sampling frequency
%w_max=pi*fs;
w_max=fs*pi;
N=14001*5; % number of samples;
t_range=(0:delta_t:(N-1)*delta_t);
t=((-250*delta_t*1e3):delta_t*1e3:(250*delta_t*1e3)); % plot range  for 50ms
figure;
%
CF1_plot=[CF1((length(CF1)-249):length(CF1)),CF1(1:251)];
CF2_plot=[CF2((length(CF2)-249):length(CF2)),CF2(1:251)];
CF3_plot=[CF3((length(CF3)-249):length(CF3)),CF3(1:251)];
CF4_plot=[CF4((length(CF4)-249):length(CF4)),CF4(1:251)];
plot(t,(CF1_plot)); hold on;
%plot(t,(CF2_plot)); 
%plot(t,(CF3_plot));
%plot(t,(CF4_plot));
title('Time domain Correlation Function C(\bf m1,\bf m2,\tau), OP= 90, S1&S2 are located in parallel');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m1,\bf m2,\tau)'); 
legend('OP=90,distance between m1&s1, m2&s2: ~2mm','distance between m1&s1~2mm, m2&s2~4mm');
hold off;

figure;
plot(t,(CF2_plot));
title('Time domain Correlation Function C(\bf m1,\bf m2,\tau),OP=0, S1 & S2 located inline');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m1,\bf m2,\tau)'); 
legend('OP=0,distance between m1&s1,m2$s2: ~2mm');

figure;
plot(t,(CF6_plot));
title('Time domain Correlation Function C(\bf m1,\bf m2,\tau),OP=45, S1 & S2 located in parallel');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m1,\bf m2,\tau)'); 
legend('OP=45, distance between m1&s1, m2&s2: ~2mm' );

figure;
plot(t,(CF4_plot));
title('Time domain Correlation Function C(\bf m1,\bf m2,\tau),OP_s1-OP_s2=90');
axis square;
grid on;
xlabel('t(ms)');
ylabel('C(\bf m1,\bf m2,\tau)'); 
legend('OP_diff=90, distance between m1&s1, m2&s2: ~2mm' );

%==========================================================================
%This part is for generating the contour plot for varying the 
%coordinates of M2. 
%{
%!!!VERY SLOW BY USING DOUBLE LOOP!!!!
m2_x=(-1:0.1:5)*1e-3;
m2_y=fliplr(-1:0.1:5)*1e-3;
m1=[1.2e-3 1.5e-3];%m measurement point #1
s2=[(2.7e-3+1e-10) (3.5e-3+1e-10)];%m source piont #2
s1=[(1.2e-3+1e-10) (3.5e-3+1e-10)];%m  source point #1
l=length(m2_x);
CF_m2=ones(l);
for mm=1:l
    for ee=1:l
        m2=[m2_x(ee) m2_y(mm)];
        CF_temp=plot_temporal_correlation_fn(s1_s,s2_s,m1,m2);
        CF_m2(mm,ee)=CF_temp(1);
    end
end
CF_m2_plot=CF_m2;
[nrow ncol]=size(CF_m2_plot);
CF_m2_log_plot=zeros(nrow);
for nn=1:nrow
    for mm=1:ncol
        CF_m2_log_plot(nn,mm)=log10(CF_m2_plot(nn,mm));
        if imag(CF_m2_log_plot(nn,mm))~=0
            CF_m2_log_plot(nn,mm)=-real(CF_m2_log_plot(nn,mm));
        end
    end
end
%{
[M,I] = max(CF_m2_log_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_log_plot),I);
CF_m2_log_plot(I_row, I_col)=NaN;
[M,I] = max(CF_m2_log_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_log_plot),I);
CF_m2_log_plot(I_row, I_col)=NaN;
figure;
contourf(m2_x,m2_y,CF_m2_log_plot,'LineStyle','none');
axis square;
xlabel('x [m]');
ylabel('y[m]');
colormap('Bone');
colorbar;
hold on;
plot(m1(1),m1(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m1(1),m1(2),'\bf M1','FontSize',9);
plot(s1(1),s1(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(s1(1),s1(2),'\bf S1','FontSize',9);
plot(s2(1),s2(2),'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(s2(1),s2(2),'\bf S2','FontSize',9);
plot([-1.0e-3, 5.0e-3], [1e-3, 1e-3],'k','LineStyle','--','LineWidth',0.5);
plot([1e-3,1e-3],[-1.0e-3, 5.0e-3],'k','LineStyle','--','LineWidth',0.5);
plot([3e-3,3e-3],[-1.0e-3, 5.0e-3],'k','LineStyle','--','LineWidth',0.5);
plot([-1.0e-3, 5.0e-3], [3e-3, 3e-3],'k','LineStyle','--','LineWidth',0.5);
hold off;

%}

[M,I] = max(CF_m2_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_plot),I);
CF_m2_plot(I_row, I_col)=NaN;
[M,I] = max(CF_m2_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_plot),I);
CF_m2_plot(I_row, I_col)=NaN;

figure;
contourf(m2_x*1e3,m2_y*1e3,CF_m2_plot,'LineStyle','none');
axis square;
xlabel('x [mm]');
ylabel('y[mm]');
colormap('Bone');
colorbar;
hold on;
plot(m1(1)*1e3,m1(2)*1e3,'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m1(1)*1e3,m1(2)*1e3,'\bf M1','FontSize',9);
plot(s1(1)*1e3,s1(2)*1e3,'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(s1(1)*1e3,s1(2)*1e3,'\bf S1','FontSize',9);
plot(s2(1)*1e3,s2(2)*1e3,'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(s2(1)*1e3,s2(2)*1e3,'\bf S2','FontSize',9);
plot([-1.0, 5.0], [1, 1],'k','LineStyle','--','LineWidth',0.5);
plot([1,1],[-1.0, 5.0],'k','LineStyle','--','LineWidth',0.5);
plot([3,3],[-1.0, 5.0],'k','LineStyle','--','LineWidth',0.5);
plot([-1.0, 5.0], [3, 3],'k','LineStyle','--','LineWidth',0.5);
hold off;
%}
%==========================================================================

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
%k=[0 0;K_lowest 0; 0 K_lowest;0 2*K_lowest;K_lowest K_lowest;K_lowest 2*K_lowest;K_lowest 3*K_lowest;];
%k=[0 0;K_lowest 0; 0 K_lowest;K_lowest K_lowest;K_lowest 2*K_lowest;];
k=[0 0;K_lowest 0; 2*K_lowest 0; 3*K_lowest 0;0 K_lowest;0 2*K_lowest;0 3*K_lowest; 0 4*K_lowest;K_lowest K_lowest;2*K_lowest K_lowest;3*K_lowest K_lowest;K_lowest 2*K_lowest;2*K_lowest 2*K_lowest;3*K_lowest 2*K_lowest;K_lowest 3*K_lowest;2*K_lowest 3*K_lowest;3*K_lowest 3*K_lowest;K_lowest 4*K_lowest; 2*K_lowest 4*K_lowest;3*K_lowest 4*K_lowest;K_lowest 5*K_lowest; 2*K_lowest 5*K_lowest;3*K_lowest 5*K_lowest;];
%k=[0,0;K_lowest 0;2*K_lowest 0;3*K_lowest 0; 0 K_lowest;0 2*K_lowest;0 3*K_lowest;K_lowest K_lowest;2*K_lowest K_lowest;K_lowest 2*K_lowest;2*K_lowest 2*K_lowest;K_lowest 3*K_lowest;2*K_lowest 3*K_lowest;];
r=m1;
r_=m2;
r2=s2;
r1=s1;

R1=r-r1; %m m1-s1 
R2=r_-r1;%m m2-s2
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

%==========================================================================
% This function returns the Transfer Function T_ms(m,s)
% Inputs: number of K terms, distance |m-s|, and the angular frenqucy range 
function T_es=TransferFn(k,R,w_)
alpha_em= 80; % s^-1
beta_em= 800; %s^-1
alpha_es= 80; % s^-1
beta_es= 800; %s^-1
alpha_ei= 80; % s^-1
beta_ei= 800; %s^-1

r_em= 2e-3; %m
r_es=0.3e-3; %m
r_ei=0.2e-3; %m
gamma_em= 500; %s^-1
G_em=6.9;
G_ei=-15;
G_es=1.7;
r_ei=0.2e-3;%m
tau_es=43e-3; % propagation delay in seconds 
a=2e-3; %m, width of the unit cell;
K_lowest=(2*pi)/a;
rho_e=4200; % V^-1s^-1
delta_t=1e-3; % time resolution in seconds;
fs=1/delta_t; %sampling frequency
%w_max=pi*fs;
w_max=fs*pi;
N=14001; % number of samples;
delta_w=(2*pi)/(N*delta_t); % frequency resolution
% store C_K as matrix
c_K_matrix=[41244.1856495544 12746.5721853508 -1463.04707664718 -1321.34431462285;...
           -27113.5331506906-1i*670.842953555454 7000.27956643087+1i*173.200895433239 838.879870013114+1i*20.7555631554973 -280.500858158204-1i*6.94015136706328;...
           1839.10736539701+1i*91.0621226930134 -2490.13851403829-1i*123.297477436290 255.202324339401+1i*12.6361656789520 1173.99177568078+1i*58.1293866410856;...
           7787.59959578219+1i*578.987653199529 -4294.35285008329-1i*319.273897957861 -587.122087329027-1i*43.6509909508389 825.361504438736+1i*61.3634682444432;...
           -671.584244229824-1i*66.6695053516027 440.553076476115+1i*43.7345812415140 104.624189682128+1i*10.3862516636601 -527.310970708370-1i*52.3472101760184;...
           -4499.35246734796-1i*559.356528942585 2450.90293658354+1i*304.694635357301 265.468369151002+1i*33.0028524304251 -683.141790987713-1i*84.9277365477752; ];
       %scaling factor
       s_factor=127; % s_factor used in NeuroEng poster result
       c_K_matrix=c_K_matrix./(s_factor);
      
w=w_; % angular frequency variable;
T_es=zeros(size(w)); % initialise T_es
s=size(k);
%use for loop to go through every spatial mode K
 for p=1:s(1)
     K=k(p,:);
     K_mag=sqrt(K(1)^2+K(2)^2);
     R_mag=sqrt((R(1))^2+(R(2))^2);
     %===========================
     % find G_hat and omega for one particular spacial mode K 
        G_ei_hat=G_ei/((K_mag^2*r_ei^2)+1);
        omega_square=(gamma_em*(2*alpha_em*beta_em*(1-G_ei_hat)+gamma_em*(alpha_em+beta_em)))/(alpha_em+beta_em+2*gamma_em);
        omega=[sqrt(omega_square),-sqrt(omega_square)];% each K has two corresponding omega values(i.e. +/- omega);
        frq=omega/(2*pi);
        G_hat=((1-(omega_square/(gamma_em^2)))*(1-(omega_square/(alpha_em*beta_em))-G_ei_hat))-(((2*omega_square)/gamma_em)*((1/alpha_em)+(1/beta_em)));
        
     %==============================
     %===============================
     % Find J_es, J_ei,J_em value corresponding to poles: (K,omega), (K,-omega)
     L_es_omega=((1-((1i*omega)/alpha_es)).^-1).*((1-((1i*omega)/beta_es)).^-1);
     L_ei_omega=((1-((1i*omega)/alpha_ei)).^-1).*((1-((1i*omega)/beta_ei)).^-1);
     L_em_omega=((1-((1i*omega)/alpha_em)).^-1).*((1-((1i*omega)/beta_em)).^-1);
     nu_es_omega=G_es/rho_e;
     nu_ei_omega=G_ei/rho_e;
     J_es_omega=(G_es.*L_es_omega.*exp(1i.*omega.*tau_es))./(1+((K_mag^2)*(r_es^2)));
     J_ei_omega=(G_ei)./(1+((K_mag^2)*(r_ei^2)));
     J_em_omega=L_em_omega*G_hat;
     %=======================================================================
     % define J_ei, J_em, J_es as a function of angular frequency range w:
     % w=[-w_max, w_max] w_max=2*pi*(fs/2); 
     % G-hat value varies with w, for one specific spacial mode K.
    G_hat_w=((1-((1i*w)/alpha_em)).*(1-((1i*w)/beta_em))-G_ei_hat).*((1-((1i*w)/gamma_em)).^2);
    L_em_w=((1-((1i*w)/alpha_em)).^-1).*((1-((1i*w)/beta_em)).^-1);
    L_ei_w=L_em_w;
    L_es_w=L_em_w;
    J_ei_w=(G_ei)./(1+((K_mag^2)*(r_ei^2)));
    J_em_w=L_em_w.*G_hat_w;
    J_es_w=(L_es_w.*G_es.*exp(1i.*w.*tau_es))./(1+((K_mag^2)*(r_es^2)));
     % for each K (i.e. K=(Kx,0), or K=(0,Ky)),it has 4 combinations of (K,omega)
     % that is(K,omega),(K,-omega), (-K,omega), (-K,_omega);
     % To=To(K,omega)+To(K,-omega)+To(-K,omega)+To(-K,-omega);
    if(K(1)==0&&K(2)==0)% K=(0,0)
        c_K_0=c_K_matrix(K(1)+1,K(2)+1);
         To...  %To value when K=(0,0),
        =(J_es_omega.*J_em_omega.*c_K_0)./((1-J_ei_omega).^2);
    q_square=(((1-1i*w/gamma_em).^2)+((J_em_w.*c_K_0)./(1-J_ei_w)))./(r_em^2);
    q=sqrt(q_square);
     %q=sqrt(q_square)*0.1;
    T_es_temp=((2*pi*r_em^2)^-1).*exp(1i.*(dot(K,R))).*To(1).*besselk(1,q.*R_mag);
    T_es_temp=T_es_temp+conj(T_es_temp); % including the compled conj. of (K=(0,0),-omega)
    elseif (K(1)==0 || K(2)==0)% K=(Kx,0) or (0,Ky)
        if(K(1)==0)%m-row index; n-column index;
            n=1;m=K(2)/K_lowest +1;
            c_K=c_K_matrix(m,n);
        elseif (K(2)==0)
             m=1;n=K(1)/K_lowest +1;
            c_K=c_K_matrix(m,n);
        end
    To_1=(J_es_omega(1).*J_em_omega(1).*c_K)./(1-J_ei_omega(1)).^2; %To_(K,omega)
    To_2=conj(To_1); %To_(K,-omega)
    % q_square value corresponds angular frequency range: w  
    q_square_1=(((1-1i*w/gamma_em).^2)+((J_em_w.*c_K)./(1-J_ei_w)))./(r_em^2);
    q_1=sqrt(q_square_1);
     %q_1=sqrt(q_square_1)*0.1;
    %T_es=T_es(K,omega)+T_es(K,-omega)+T_es(-K,omega)+T_es(-K,-omega)
    T_es_temp_1... % T_es(K,omega)
        = ((2*pi*r_em^2)^-1).*exp(1i.*(dot(K,R))).*To_1.*besselk(1,q_1.*R_mag);
     T_es_temp_1=T_es_temp_1+conj(T_es_temp_1); % add the T_es(-K,-omega)
    T_es_temp_2... % T_es(K,-omega)
        = ((2*pi*r_em^2)^-1).*exp(1i.*(dot(K,R))).*To_2.*besselk(1,q_1.*R_mag);
    T_es_temp_2=T_es_temp_2+conj(T_es_temp_2); % add the T_es(-K,omega)
    T_es_temp=(T_es_temp_1+T_es_temp_2); 
    else % case when K=(+/-Kx, +/-Ky), totally 8 combinations of K and omega
        n=(K(1)/K_lowest +1);m=(K(2)/K_lowest +1);%m-row index; n-column index;
        c_K=c_K_matrix(m,n);
    To_1=(J_es_omega(1).*J_em_omega(1).*c_K)./(1-J_ei_omega(1)).^2; %To_((Kx,Ky),omega)
    To_2=conj(To_1); %To_((Kx,Ky),-omega)
    To_3=To_1;  %To_((-Kx,Ky),omega)
    To_4=To_2;  %To_((-Kx,Ky),-omega)
   
    q_square_1=(((1-1i*w/gamma_em).^2)+((J_em_w.*c_K)./(1-J_ei_w)))./(r_em^2);
    q_1=sqrt(q_square_1);
    T_es_temp_1... % T_es((Kx,Ky),omega)
        = ((2*pi*r_em^2)^-1).*exp(1i.*(dot(K,R))).*To_1.*besselk(1,q_1.*R_mag);
    T_es_temp_1= T_es_temp_1+conj(T_es_temp_1); %adding T_es_temp_1*,(i.e. T_es((-Kx,-Ky),-omega))
    T_es_temp_2... % T_es((Kx,Ky),-omega)
        = ((2*pi*r_em^2)^-1).*exp(1i.*(dot(K,R))).*To_2.*besselk(1,q_1.*R_mag);
    T_es_temp_2= T_es_temp_2+conj(T_es_temp_2);% adding T_es_temp_2*,T_es((-Kx,-Ky),omega)
    T_es_temp_3... % T_es((-Kx,Ky),omega)
        = ((2*pi*r_em^2)^-1).*exp(1i.*(dot([-K(1) K(2)],R))).*To_3.*besselk(1,q_1.*R_mag);
    T_es_temp_3= T_es_temp_3+conj(T_es_temp_3);% adding T_es_temp_3*,T_es((Kx,-Ky),-omega)
    T_es_temp_4... % T_es((-Kx,Ky),-omega)
        = ((2*pi*r_em^2)^-1).*exp(1i.*(dot([-K(1) K(2)],R))).*To_4.*besselk(1,q_1.*R_mag);
    T_es_temp_4= T_es_temp_4+conj(T_es_temp_4);% adding T_es_temp_4*,T_es((Kx,-Ky),omega)
    
    T_es_temp=(T_es_temp_1+T_es_temp_2+T_es_temp_3+T_es_temp_4);
    end
    
    T_es=T_es+T_es_temp;
 end
 

end