function [r, info] = acproj(x, o)
% AC-feasible projection of a fixed dispatch: AC-OPF minimizing sum (PG - PG_hist)^2
% subject to the AC power flow equations, gen P/Q limits and bus voltage limits.
define_constants;
pg = x.gen(:,PG); ng = size(x.gen,1);
x.gencost = [2*ones(ng,1) zeros(ng,2) 3*ones(ng,1) ones(ng,1) -2*pg pg.^2];
x.gen(:,PMIN) = min(x.gen(:,PMIN), pg); x.gen(:,PMAX) = max(x.gen(:,PMAX), pg);
x.branch(:,RATE_A) = 0;   % NYgrid ratings are sparse/placeholder; stability limits come from the AC equations
if isfield(x,'genfuel'), x = rmfield(x,'genfuel'); end
r = runopf(x, mpoption(o,'opf.ac.solver','MIPS','mips.max_it',300));
dp = r.gen(:,PG) - pg;
info.ok = r.success; info.redisp = sum(abs(dp))/2; info.maxdp = max(abs(dp));
info.total = sum(pg); info.vmin = min(r.bus(:,VM)); info.vmax = max(r.bus(:,VM));
end
