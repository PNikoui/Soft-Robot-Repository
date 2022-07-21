clear all;close all;clc;
% Create plots for the relationship between the digits and the pressure in
% the pnuematic actuators
MAT = csvread('PressureDigit_Relationship.csv');


Digits = MAT(1:2:end,1);
Voltage = MAT(1:2:end,2);  % V = Digit*5/255
Pressure = MAT(1:2:end,3);
Real_Pressure = MAT(1:2:end,4);


plot(Digits,Pressure,'k--*','DisplayName','Pressure','linewidth',2)
hold on
plot(Digits,Real_Pressure,'b--*','DisplayName','Real Pressure','linewidth',2)
xlabel('Digit Chamber Input')
xlim([33,246])
ylabel('Pressure [Bar]')
ylim([0.17,1.43])
yyaxis right 
plot(Digits,Voltage,'r--*','DisplayName','Voltage','linewidth',2)
ylabel('Voltage [V]')
ylim([0.64,4.82])
grid on 
legend('Location','southeast')
set(gca, 'LooseInset', get(gca,'TightInset'))
saveas(gcf,'Pressure_Digit_Relationship.pdf')