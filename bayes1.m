function latents = bayes1(data, param)

    % simple Bayesian learning (assumes binary rewards)
    % learn posterior over P(r|s,a) for each (s,a) pair, with Beta prior
    % with optional decay of Beta parameters
    % reset Beta parameters for each session
    %

    if ~exist('param', 'var')
        alpha0 = 1; % beta param
        beta0 = 1; % beta param
        tau = 1; % softmax temperature
        decay = 0.1; % decay rate
    else
        alpha0 = param(1);
        beta0 = param(2);
        tau = param(3);
        decay = param(4);
    end

    S = max(data.cue); % # states / cues
    A = 2; % # actions

    for i = 1:length(data.cue)
        s = data.cue(i);

        if i == 1 || (i > 1 && data.sesh(i - 1) ~= data.sesh(i))
            % reset for each session
            alpha = ones(S,A) * alpha0;
            beta = ones(S,A) * beta0;
        end

        [Q, ~] = betastat(alpha, beta); % Q values = expected P(r|s,a) = mean of beta for (s,a)

        p = softmax(Q(s,:), tau);
        a = find(mnrnd(1, p));

        reward = data.r(i,a);
        PE = reward - Q(s,a);

        alpha = alpha * (1 - decay) + alpha0 * decay;
        beta = beta * (1 - decay) + beta0 * decay;
        if reward == 1
            alpha(s,a) = alpha(s,a) + 1;
        else
            beta(s,a) = beta(s,a) + 1;
        end

        latents.allQ(:,:,i) = Q;
        latents.alpha(:,:,i) = alpha;
        latents.beta(:,:,i) = beta;
        latents.Q(i,:) = Q(s,:);
        latents.reward(i) = reward;
        latents.PE(i) = PE;
        latents.p(i,:) = p;
        latents.a(i) = a;
    end

end

function p = softmax(Q, tau)
    p = exp(Q / tau);
    p = p / sum(p);
end




