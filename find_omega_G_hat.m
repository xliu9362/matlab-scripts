
a = 1e-3; %m
gamma_em=500;
alpha_em=80;
beta_em=800;
alpha_=alpha_em/gamma_em;
beta_=beta_em/gamma_em;
K=(2*pi) / a;
r_ei=0.2e-3;
G_ei=-15;
G_ei_hat=G_ei/((K^2*r_ei^2)+1);
omega_square=(gamma_em*(2*alpha_em*beta_em*(1-G_ei_hat)+gamma_em*(alpha_em+beta_em)))/(alpha_em+beta_em+2*gamma_em);
omega=sqrt(omega_square)
frq=omega/(2*pi)
G_hat=((1-(omega_square/(gamma_em^2)))*(1-(omega_square/(alpha_em*beta_em))-G_ei_hat))-(((2*omega_square)/gamma_em)*((1/alpha_em)+(1/beta_em)))
