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