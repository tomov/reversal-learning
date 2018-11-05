
%gen_data('data.csv', 3, 2, 10, 50);

clear all;
data = load_data('data.csv');

for s = 1:length(data)
    %latents(s) = qlearn(data(s));
    %latents(s) = bayes1(data(s));
    latents(s) = bayes2(data(s));
end

subj = 1;
seshs = 1;
blocks = 1:5;
which = ismember(data(subj).sesh, seshs) & ismember(data(subj).block, blocks);

figure;
subplot(3,1,1);
plot(data(subj).choice(which) - 1, '*');
hold on;
plot(latents(subj).a(which) - 1);
plot(latents(subj).p(which, 1));
hold off;
set(gca, 'YLim', [-0.2 1.2]);
legend({'subject choices', 'model choices', 'model P(choose 1)'});
title('choices');

subplot(3,1,2);
plot(latents(subj).PE(which));
legend({'PE'});
title('prediction error');

subplot(3,1,3);
plot(latents(subj).Q(which, :));
legend({'Q(1)', 'Q(2)'});
title('action values');

fprintf('Total reward: %.2f\n', sum(latents(subj).reward(which)));
