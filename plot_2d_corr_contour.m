%==========================================================================
%This part is for generating the contour plot for varying the 
%coordinates of M2. 
%
%!!!VERY SLOW BY USING DOUBLE LOOP!!!!
m2_x=(-3:0.1:7)*1e-3;
m2_y=fliplr(-1:0.1:9)*1e-3;
m1=[1.2e-3 1.5e-3];%m measurement point #1
s2=[(2.7e-3) (3.5e-3)];%m source piont #2
s1=[(1.2e-3) (3.5e-3)];%m  source point #1
l=length(m2_x);
CF_m2=ones(l);
for mm=1:l
    for ee=1:l
        m2=[m2_x(ee) m2_y(mm)];
        CF_temp=temporal_correlation_fn(s1,s2,m1,m2);
        CF_m2(mm,ee)=CF_temp(1);
    end
end
CF_m2_plot=real(CF_m2);


[M,I] = max(CF_m2_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_plot),I);
CF_m2_plot(I_row, I_col)=NaN;
[M,I] = max(CF_m2_plot(:));
[I_row, I_col] = ind2sub(size(CF_m2_plot),I);
max_value=CF_m2_plot(I_row, I_col);
figure;
contourf(m2_x*1e3,m2_y*1e3,CF_m2_plot./max_value,'LineStyle','none');
axis square;
xlabel('x [mm]');
ylabel('y[mm]');
colormap('redblue');
colorbar;
hold on;
plot(m1(1)*1e3,m1(2)*1e3,'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(m1(1)*1e3,m1(2)*1e3,'\bf M1','FontSize',9);
plot(s1(1)*1e3,s1(2)*1e3,'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(s1(1)*1e3,s1(2)*1e3,'\bf S1','FontSize',9);
plot(s2(1)*1e3,s2(2)*1e3,'marker','x',  'LineWidth',3, 'MarkerSize',10, 'MarkerEdgeColor','k');
text(s2(1)*1e3,s2(2)*1e3,'\bf S2','FontSize',9);
plot([-3.0, 7.0], [1, 1],'k','LineStyle','--','LineWidth',0.5);
plot([-1,-1],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([1,1],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([3,3],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([5,5],[-1.0, 9.0],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 7.0], [3, 3],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 7.0], [5, 5],'k','LineStyle','--','LineWidth',0.5);
plot([-3.0, 7.0], [7, 7],'k','LineStyle','--','LineWidth',0.5);
hold off;


%