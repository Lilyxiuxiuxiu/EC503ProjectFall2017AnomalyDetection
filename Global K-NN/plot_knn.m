figure(1)
plot(pg_precision);
hold on
plot(kdd_precision);
plot(sanota_precision);
plot(versic_precision);
plot(virigi_precision);
hold off
xlabel('k');
ylabel('value of precision');
title('precision plot of all five data sets');
legend('Pen-Global', 'Kdd', 'iris sanota', 'iris versic', 'iris virigi');

figure(2)
plot(pg_recall);
hold on
plot(kdd_recall);
plot(sanota_recall);
plot(versic_recall);
plot(virigi_recall);
hold off
xlabel('k');
ylabel('value of recall');
title('recall plot of all five data sets');
legend('Pen-Global', 'Kdd', 'iris sanota', 'iris versic', 'iris virigi');

figure(3)
plot(pg_fsocre);
hold on
plot(kdd_fscore);
plot(sanota_fscore);
plot(versic_fscore);
plot(virigi_fscore);
hold off
xlabel('k');
ylabel('value of fscore');
title('fscore plot of all five data sets');
legend('Pen-Global', 'Kdd', 'iris sanota', 'iris versic', 'iris virigi');