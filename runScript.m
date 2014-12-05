[GenErr1 RandErr1] = DHM2();
[GenErr2 RandErr2] = DHM2();
[GenErr3 RandErr3] = DHM2();

GenErr = [GenErr1'; GenErr2'; GenErr3'];
RandErr = [RandErr1'; RandErr2'; RandErr3'];

figure(1);
errorbar(mean(GenErr), std(GenErr));
hold on;
errorbar(mean(RandErr), std(RandErr));
hold off;