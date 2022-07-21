function [v1,v2,v3,v4,v5,v6,v7,v8,s1,s2,s3,s4,s5,s6] = ConditionCheck(v1,v2,v3,v4,v5,v6,v7,v8,s1,s2,s3)
%ConditionCheck Checks the pre-defined motor/chamber conditions for coupled
% movement of the two
%   The conditions are defined as:
%
% i. The user has to inflate a chamber (P1), it is recommended to release the two cables that
% are on each side (C2-C3)
%
% ii. The user has to inflate two chambers (P1-P2), it is recommended to release the cable in
% between (C3)
%
% iii. The user has to inflate three chambers (P1-P2-P3), it is recommended to release all the
% three cables (C1-C2-C2). These mechanisms avoid destroying the fluidic actuators, balancing the forces in each
% actuators direction. Once this motion has been done, the stiffness can be tuned by acting on the single actuators.
%
% iv. The user has to pull a cable (C1), it is recommended to release the other two cables (C2-C3)
%
% v. The user has to pull two cables (C1-C2), it is recommended to release the third cable
% (C3).

% sMax = 900;
sZero = 450;
sMin = 300;
% vMin = 0;
vMax = 255;

v4 = 0; v5= 0;v6=0;v7=0;v8=0;

% v.
if s1<sZero && s2<sZero
    s3 = sZero + sZero*((s1 + s2)/(sMin + sMin));
    % I added the inverse relationship to be safe
    v3 = v3 - vMax*((s1 + s2)/(sMin + sMin));
    
elseif s2<sZero && s3<sZero
    s1 = sZero + sZero*((s2 + s3)/(sMin + sMin));
    % Inverse
    v1 = v1 - vMax*((s2 + s3)/(sMin + sMin));
    
elseif s3<sZero && s1<sZero
    s2 = sZero + sZero*((s3 + s1)/(sMin + sMin));
    % Inverse
    v2 = v2 - vMax*((s3 + s1)/(sMin + sMin));
     
% iv.
elseif s1<sZero 
    s3 = sZero + sZero*(s1/sMin);
    s2 = sZero + sZero*(s1/sMin);
    % Inverse
    v3 = v3 - vMax*(s1/sMin);
    v2 = v2 - vMax*(s1/sMin);
    
elseif s2<sZero 
    s3 = sZero + sZero*(s2/sMin);
    s1 = sZero + sZero*(s2/sMin);
    % Inverse
    v1 = v1 - vMax*(s2/sMin);
    v3 = v3 - vMax*(s2/sMin);
    
elseif s3<sZero
    s1 = sZero + sZero*(s3/sMin);
    s2 = sZero + sZero*(s3/sMin);
    % Inverse
    v1 = v1 - vMax*(s3/sMin);
    v2 = v2 - vMax*(s3/sMin);

% iii.
elseif v1>0 && v2>0 && v3>0
    s1 = sZero + sZero*((v1 + v2 + v3)/(vMax + vMax + vMax));
    s2 = sZero + sZero*((v1 + v2 + v3)/(vMax + vMax + vMax));
    s3 = sZero + sZero*((v1 + v2 + v3)/(vMax + vMax + vMax));
    
% ii.
elseif v1>0 && v2>0
    s3 = sZero + sZero*((v1 + v2)/(vMax + vMax));
elseif v2>0 && v3>0
    s1 = sZero + sZero*((v2 + v3)/(vMax + vMax));
elseif v3>0 && v1>0
    s2 = sZero + sZero*((v3 + v1)/(vMax + vMax));

% i.

elseif v1>0
    s2 = sZero + sZero*(v1/255);
    s3 = sZero + sZero*(v1/255);
    
elseif v2>0
    s3 = sZero + sZero*(v2/255);
    s1 = sZero + sZero*(v2/255);
    
elseif v3>0
    s1 = sZero + sZero*(v3/255);
    s2 = sZero + sZero*(v3/255);
    
end

%safety check to keep within low range
if s1<300
    s1=300;
end
if s2<300
    s2=300;
end
if s3<300
    s3=300;
end
% if s4<230
%     s4=230;
% end
% if s5<230
%     s5=230;
% end
% if s6<230
%     s6=230;
% end

%safety check to keep within high range
if s1>900
    s1=900;
end
if s2>900
    s2=900;
end
if s3>900
    s3=900;
end
% if s4>900
%     s4=900;
% end
% if s5>900
%     s5=900;
% end
% if s6>900
%     s6=900;
% end

end

