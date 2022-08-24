clear all; close all; clc
clear classes
%%
% Initialization of Soft Robot
Ts = 1;
M = 0;
NTimeSteps = 250; 
NObservations = 50;  %% How many waypoints on the trajectory
NStepPerTarget = NTimeSteps/NObservations;
% Tx = zeros(N);Ty = zeros(N);
xx = linspace(0,2*pi,NObservations);yy = linspace(0,2*pi,NObservations);
% Define target of end effector
r = 10; 
Txi = r*cos(xx);
Tyi = r*sin(yy);
Tx = 1;
Ty = 1;
Tz = 2;

t_iterO = 1;
% Initial end effector positions
x_i = [[1]
       [1]
       [1]];
% System matrix
A = [[0, 0, 0]
     [0, -1.4044, 0.7363]
     [0, 0, -6.1535]];
% Actuator matrix
B1 = [[-1.6909,2.0717,-0.5338]
      [1.4754, 1.1778,-2.9646]
      [0.3089,0.5663,0.0359]];
% Measurement matrix
C = eye(3); % How well do the sensors capture the position?

NStates = 3;
% U_1 = prbs(3,NTimeSteps); U_2 = zeros(1,NTimeSteps); U_3 = zeros(1,NTimeSteps);
USize = 1000;
InputData.time = [linspace(0,NTimeSteps,USize)]';
InputData.signals.values = 1.4*rand(3,USize)';
Etol = 10; t_iterO =1;
Var = 5; Mu = 0;
% simData = cell(1,NStates,NObservations);
a1 = 5;
b = 2;

%% Open UDP Communication Socket
pyenv
SetPath('UDP.py')
SetPath('SNN_model.py')
SetPath('SNN_in_Py.py')


% %% Read CSV UDP Data
% 
% UDP_Data = readtable("UDPOutout.csv");
% plot(UDP_Data.Variables,linewidth=2)
% xlabel('Time [s]')
% ylabel('Position [cm]')
% legend('x','y','z','location','southwest')
% 
% %% Open Loop
% 
% SIM = sim('SR_SNN.slx');
% simDataOpenLoop = SIM.ModelOutput.Data(2:end,:);
% simInputsOpenLoop = [U_1;U_2;U_3]';
% 
% 
% %% Save CSV for SNN Training
% 
% csvwrite('SNNTrainingData.csv',simDataOpenLoop);
% csvwrite('SNNTargetData.csv',simInputsOpenLoop);


%% Reload Python modules 
reloader('SNN_model')
reloader('SNN')
reloader('SNN_in_Py')

% %% Run SNN Model 
% py.SNN_in_Py.Network().Run_model([10,10,10])
%% Simulation TEST
% Loop to iterate through input data

% Tx = Txi(i); Ty = Tyi(i);
t_iter = 1;
% SIM = sim('SR_SNN.slx');

SIM = sim('SR_SNN_Extrinsic_code.slx');

% simData{:,:,i} = {SIM.ModelOutput.Data};


% %% Simulation 
% % Loop to iterate through input data
% for i = 1:NObservations
%     M = M + 10;
%     S1 = 450+M;     S2 = 450+M;     S3 = 450+M;     S4 = 450+M;     S5 = 450+M;     S6 = 450+M;
% 
%     Tx = Txi(i); Ty = Tyi(i);
%     %     xpos = 0; ypos = 0; zpos = 0;
%     %
%     %     % Calculate Control action from SNN model
%     %     [V1,V2,V3,S1,S2,S3,S4,S5,S6] = RunPython(xpos,ypos,zpos);
%     %
%     %     % Check Conditions for motor/chamber movement
%     %     [V1,V2,V3,S1,S2,S3,S4,S5,S6] = ConditionCheck(V1,V2,V3,V4,V5,V6,V7,V8,S1,S2,S3,S4,S5,S6);
% 
%     SIM = sim('SR_SNN.slx');
% 
%     simData{:,:,i} = {SIM.ModelOutput.Data};
% %     disp(M)
% end
% 
% 
% %% Plotting
% 
% Targets = [];
% DATA = [];
% for i=1:NObservations
% 
%     % Targets
%     Ti = Txi(i)*ones(size(cell2mat(simData{:,:,i}),1),1);
%     Targets = [Targets;Ti];
% 
%     % Data
%     Di = cell2mat(simData{:,:,i});
%     DATA = [DATA;Di];
% end
%%
% Generic set up 
% clear, close, clc

% Simulation of the model implemented in Simulink
open('SR_SNN_Extrinsic_code.slx')
Ts = 1e-3;
for i = 1:6
    % Specific step value
    STEP = 20*i;
        
    % Simulate the system
    sim('SR_SNN_Extrinsic_code.slx')
           
    % Plot the results
    figure(1)
    title("Step responses")
    plot(tout, theta*180/pi, 'b')
    hold on, grid on
    plot(tout, theta_d*180/pi, 'r')
    xlabel('time $t$', 'Interpreter','latex','FontSize',12)
    ylabel('Rotation angle [deg]', 'Interpreter','latex','FontSize',12)
    sprintf('The error at %d is %d.',20*i, (theta(end) - theta_d(end))*180/pi)
end

%%
Pos = SIM.ModelOutput.data;
%% 1-D
plot(Pos(:,1),'-',LineWidth=2)
hold on
plot(Tx*ones(length(Pos(:,1))),'k--', LineWidth=2)
xlabel('Time [s]')
xlim([0,NTimeSteps])
ylabel('Position [mm]')
grid on
legend('x','Target','location','southeast')

%% 2-D
plot(Pos(:,1),Pos(:,2),'-',LineWidth=2)
hold on
plot(Tx,Ty,'kX',LineWidth=1.5, MarkerSize=12)
xlabel('x Position [mm]')
xlim([-2,Tx+2])
ylabel('y Position [mm]')
ylim([-2,Ty+2])
grid on
legend('x-y','Target','location','southeast')

%% 3-D
plot3(Pos(:,1),Pos(:,2),Pos(:,3), LineWidth = 1.5)
hold on
% plot3(Txi,Tyi,Tz*ones(size(Txi)))
plot3(Tx,Ty,Tz, 'kX', LineWidth = 1.5, MarkerSize = 12)
grid on
xlabel('x Position [mm]')
xlim([-2,Tx+2])
ylabel('y Position [mm]')
ylim([-2,Ty+2])
zlabel('z Position [mm]')
zlim([0,Tz+2])
legend('x-y-z','Target','location','southeast')

%% Model Identification for Comparison with Trained SNN

ID = out.Targets.Data;
OD = out.ControlInput.Data;

% U1 = get( ID, 'inputData' );
% Y1 = get( OD, 'outputData' );
DATA = iddata(OD(:,1),ID);
[sys,ic] = armax(DATA,[3 3 3 3]);

xlim([-66 85])
ylim([-3748 3766])
plot(u3,'k')
hold on
plot(y3,'r')
hold off

%% Plotting for report 1

%%%%%% Untrained Tracking plot
subplot(2,1,1)
plot(out.ContinuousPositions.Data(:,1),out.ContinuousPositions.Data(:,2))
hold on
plot(out.Targets.Data(:,1),out.Targets.Data(:,2), 'k--', LineWidth = 1.5, MarkerSize = 12)
grid on
xlabel('x Position [mm]')
% xlim([-2,Tx+2])
ylabel('y Position [mm]')
% ylim([-2,Ty+2])
legend('x-y Response','Target','location','southeast')
subplot(2,1,2)
plot(out.ContinuousPositions.time,out.ContinuousPositions.Data(:,3))
hold on 
plot(Tz*ones(max(out.ContinuousPositions.time)), 'k--', LineWidth = 1.5)
grid on
xlabel('Time [s]')
% xlim([0,length(out.ContinuousPositions.Data(:,3))])
ylabel('z Position [mm]')
% ylim([-2,Ty+2])
set(gca, 'LooseInset', get(gca,'TightInset'))
% saveas(gcf,'UntrainedTracking.pdf')

%% Plotting for report 2 

%%%%%% Untrained Tracking plot
plot(Error.time,Error.Data, LineWidth = 1.5)
grid on
xlabel('Time [s]')
% xlim([0,length(Error.time)])
ylabel('Error [mm]')
% ylim([-2,Ty+2])
legend('x Error','y Error','z Error','location','southwest')
% set(gca, 'LooseInset', get(gca,'TightInset'))
% saveas(gcf,'UntrainedErrorFormat.pdf')