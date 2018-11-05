function gen_data(filename, nsubj, nsesh, nblocks, ntrials)

    subj = [];
    sesh = [];
    block = [];
    trial = [];
    cond = [];
    cue = [];
    r1 = [];
    r2 = [];
    reward = [];
    choice = [];
    for s = 1:nsubj
        for se = 1:nsesh
            for b = 1:nblocks
                for t = 1:ntrials
                    cnd = mod(b, 2) + 1;
                    cu = randi([1 2]);
                    if (cnd == 1 && cu == 1) || (cnd == 2 && cu == 2)
                        r = [1 0];
                    else
                        r = [0 1];
                    end

                    p = r * 0.8; % choose better one 80% of the time
                    p(p == 0) = 1 - sum(p);
                    ch = find(mnrnd(1, p));

                    subj = [subj; s];
                    sesh = [sesh; se];
                    block = [block; b];
                    trial = [trial; t];
                    cond = [cond; cnd];
                    cue = [cue; cu];
                    r1 = [r1; r(1)];
                    r2 = [r2; r(2)];
                    choice = [choice; ch];
                    reward = [reward; r(ch)];
                end
            end
        end
    end

    T = table(subj, sesh, block, trial, cond, cue, r1, r2, choice, reward);
    writetable(T, filename);

