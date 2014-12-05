function val = generateDHMGraph()

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

DHMErr = [];
RErr = [];
for (i =1:10),
    [DHMErrN RErrN] = DHM2();
    sizeArr = [0 (15 - size(DHMErrN, 2))];
    DHMErrN = padarray(DHMErrN, sizeArr, 'post');
    RErrN = padarray(RErrN, sizeArr, 'post');
    
    DHMErr = [DHMErr; DHMErrN];
    RErr = [RErr; RErrN];
end

DHMstd = std(DHMErr, 0, 1);
Rstd = std(RErr, 0, 1);

DHMstd = DHMstd./sqrt(10);
Rstd = Rstd./sqrt(10);

DHMMean = mean(DHMErr);
RMean = mean(RErr);

errorbar(DHMMean, DHMstd)
hold on
errorbar(RMean, Rstd, 'r')
hold off
xlabel('Number of Queries');
ylabel('Generalization Error');
legend('CAL', 'Random');

end