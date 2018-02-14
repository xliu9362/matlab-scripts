%=========================================
%02-09-2016
%orientated elliptical Gaussian function
%==========================================
[x,y]=meshgrid(-4:0.01:4,-4:0.01:4);

%============================================================================
%OP=0
A=0;
x_0=0.1;
y_0=0.5;
f=exp(-0.5*(((((x-x_0).*cos(A)+(y-y_0).*sin(A)).^2)./7)+(((-(x-x_0).*sin(A)+(y-y_0).*cos(A)).^2)./0.5)));
figure;
contourf(x,y,f,'linestyle','none');hold on;
colormap('bone'); 
axis square;
xlabel('x(mm)','FontSize',26);
ylabel('y(mm)','FontSize',26);
title('theta(r'')=0 degree');
colorbar;
plot([-1.0, 1.0], [-1, -1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,1],[1, 1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,-1],[-1, 1.0],'r','LineStyle','-','LineWidth',0.5);
plot([1, 1], [-1, 1],'r','LineStyle','-','LineWidth',0.5);
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(x_0,y_0,'\bf r''','FontSize',30);
hold off;
basic_fn=((0.5*(cos(pi*(x-x_0))+1))).*((0.5*(cos(pi*(y-y_0))+1)));
figure;
contourf(x,y,basic_fn.*f,'linestyle','none');hold on;
colormap('bone'); 
axis square;
xlabel('x(mm)','FontSize',26);
ylabel('y(mm)','FontSize',26);
colorbar;
plot([-1.0, 1.0], [-1, -1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,1],[1, 1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,-1],[-1, 1.0],'r','LineStyle','-','LineWidth',0.5);
plot([1, 1], [-1, 1],'r','LineStyle','-','LineWidth',0.5);
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(x_0,y_0,'\bf r''','FontSize',30);
hold off;
%
%
%
%==========================================================================
%OP=45
figure;
A=pi/4;
x_0=0.5;
y_0=0.2;
f=exp(-0.5*(((((x-x_0).*cos(A)+(y-y_0).*sin(A)).^2)./7)+(((-(x-x_0).*sin(A)+(y-y_0).*cos(A)).^2)./0.5)));
contourf(x,y,f,'linestyle','none');hold on;
colormap('bone'); 
axis square;
xlabel('x(mm)','FontSize',26);
ylabel('y(mm)','FontSize',26);
title('theta(r'')=45 degree');
colorbar;
plot([-1.0, 1.0], [-1, -1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,1],[1, 1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,-1],[-1, 1.0],'r','LineStyle','-','LineWidth',0.5);
plot([1, 1], [-1, 1],'r','LineStyle','-','LineWidth',0.5);
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(x_0,y_0,'\bf r''','FontSize',30);
hold off;
basic_fn=((0.5*(cos(pi*(x-x_0))+1))).*((0.5*(cos(pi*(y-y_0))+1)));
figure;
contourf(x,y,basic_fn.*f,'linestyle','none');hold on
colormap('bone'); 
axis square;
xlabel('x(mm)','FontSize',26);
ylabel('y(mm)','FontSize',26);
colorbar;
plot([-1.0, 1.0], [-1, -1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,1],[1, 1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,-1],[-1, 1.0],'r','LineStyle','-','LineWidth',0.5);
plot([1, 1], [-1, 1],'r','LineStyle','-','LineWidth',0.5);
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(x_0,y_0,'\bf r''','FontSize',30);
hold off;
%
%
%
%==========================================================================
%OP=90
A=pi/2;
x_0=0.7;
y_0=-0.5;
figure;
f=exp(-0.5*(((((x-x_0).*cos(A)+(y-y_0).*sin(A)).^2)./7)+(((-(x-x_0).*sin(A)+(y-y_0).*cos(A)).^2)./0.5)));
contourf(x,y,f,'linestyle','none');hold on;
colormap('bone'); 
axis square;
xlabel('x(mm)','FontSize',26);
ylabel('y(mm)','FontSize',26);
title('theta(r'')=90 degree');
colorbar;
plot([-1.0, 1.0], [-1, -1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,1],[1, 1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,-1],[-1, 1.0],'r','LineStyle','-','LineWidth',0.5);
plot([1, 1], [-1, 1],'r','LineStyle','-','LineWidth',0.5);
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(x_0,y_0,'\bf r''','FontSize',30);
hold off;
basic_fn=((0.5*(cos(pi*(x-x_0))+1))).*((0.5*(cos(pi*(y-y_0))+1)));
figure;
contourf(x,y,basic_fn.*f,'linestyle','none');hold on;
colormap('bone'); 
axis square;
xlabel('x(mm)','FontSize',26);
ylabel('y(mm)','FontSize',26);
colorbar;
plot([-1.0, 1.0], [-1, -1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,1],[1, 1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,-1],[-1, 1.0],'r','LineStyle','-','LineWidth',0.5);
plot([1, 1], [-1, 1],'r','LineStyle','-','LineWidth',0.5);
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(x_0,y_0,'\bf r''','FontSize',30);hold off;
%
%
%
%==========================================================================
%OP=135
A=3*pi/4;
x_0=-0.5;
y_0=-0.7;
figure;
f=exp(-0.5*(((((x-x_0).*cos(A)+(y-y_0).*sin(A)).^2)./7)+(((-(x-x_0).*sin(A)+(y-y_0).*cos(A)).^2)./0.5)));
contourf(x,y,f,'linestyle','none');hold on;
colormap('bone'); 
axis square;
xlabel('x(mm)','FontSize',26);
ylabel('y(mm)','FontSize',26);
title('theta(r'')=135 degree');
colorbar;
plot([-1.0, 1.0], [-1, -1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,1],[1, 1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,-1],[-1, 1.0],'r','LineStyle','-','LineWidth',0.5);
plot([1, 1], [-1, 1],'r','LineStyle','-','LineWidth',0.5);
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(x_0,y_0,'\bf r''','FontSize',30);
hold off;
basic_fn=((0.5*(cos(pi*(x-x_0))+1))).*((0.5*(cos(pi*(y-y_0))+1)));
figure;
contourf(x,y,basic_fn.*f,'linestyle','none');hold on;
colormap('bone'); 
axis square;
xlabel('x(mm)','FontSize',26);
ylabel('y(mm)','FontSize',26);
colorbar;
plot([-1.0, 1.0], [-1, -1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,1],[1, 1],'r','LineStyle','-','LineWidth',0.5);
plot([-1,-1],[-1, 1.0],'r','LineStyle','-','LineWidth',0.5);
plot([1, 1], [-1, 1],'r','LineStyle','-','LineWidth',0.5);
plot(x_0,y_0,'marker','.',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(x_0,y_0,'\bf r''','FontSize',30);
hold off;






