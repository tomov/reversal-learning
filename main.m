%{
gen_data('data.csv', 3, 2, 5, 50);

%}
%{
clear all;
data = load_data('data.csv');

for s = 1:length(data)
    %latents(s) = qlearn(data(s));
    %latents(s) = bayes1(data(s));
    %latents(s) = bayes2(data(s));
end

latents = bayes2struct(data(1));

subj = 1;
seshs = 1;
blocks = 1:5;
which = ismember(data(subj).sesh, seshs) & ismember(data(subj).block, blocks);

figure;

subplot(5,1,1);
plot(data(subj).choice(which) - 1, '*');
hold on;
plot(latents(subj).a(which) - 1);
plot(latents(subj).p(which, 1));
hold off;
set(gca, 'YLim', [-0.2 1.2]);
legend({'subject choices', 'model choices', 'model P(choose 1)'});
title('choices');

subplot(5,1,2);
plot(latents(subj).PE(which));
legend({'PE'});
title('prediction error');

subplot(5,1,3);
plot(latents(subj).Q(which, :));
legend({'Q(1)', 'Q(2)'});
title('action values');
%}

subplot(5,1,4);
plot(latents(subj).P(which, 1:3));
legend({'P(e=1)', 'P(e=2)', 'P(e=3)'});
title('posterior over event types (belief state)');

subplot(5,1,5);
KL = KL_divergence(latents(subj).P(which, 1:10), latents(subj).prior(which, 1:10));
plot(KL);
legend({'KL'});
title('KL divergence between posterior and prior (belief state update)');

fprintf('Total reward: %.2f\n', sum(latents(subj).reward(which)));
