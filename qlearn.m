function latents = qlearn(data, param)

    if ~exist('param', 'var')
        alpha = 0.1; % learning rate
        tau = 1; % softmax temperature
        Q0 = 0;
    else
        alpha = param(1);
        tau = param(2);
        Q0 = param(3);
    end

    S = max(data.cue); % # states / cues
    A = 2; % # actions

    Q = ones(S,A) * Q0;
    for i = 1:length(data.cue)
        s = data.cue(i);

        if i > 1 && data.sesh(i - 1) ~= data.sesh(i)
            Q = ones(S,A) * Q0; % reset after each session
        end

        p = softmax(Q(s,:), tau);
        a = find(mnrnd(1, p));

        reward = data.r(i,a);
        PE = reward - Q(s,a);
        Q(s,a) = Q(s,a) + alpha * PE;

        latents.allQ(:,:,i) = Q;
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




