function ablation(mpdir, worker, nworkers, outfile)
% Drop-one test of the AC recipe in ac_year.m: which changes to NYgrid does AC feasibility need?
% Each 2019 hour is built with NYgrid's own updateOpCond and solved under five variants:
%   full     : fixes 'JL' + extbase + acproj (the recipe)
%   noJ      : without the zone J generator split
%   noL      : without the 765 kV Massena-Marcy line
%   noE      : without extbase, so the external areas keep NYgrid's scaling with NY load
%   projOnly : acproj on NYgrid's raw operating point, no structural change at all
% Plain AC power flow without acproj is already in results/nygrid_ac_2019.csv (fix_plain_ok).
% Run from the NYgrid checkout with nygrid_full_case.patch applied; this worker takes every
% nworkers-th hour starting at hour `worker`.
addpath(genpath(mpdir)); addpath(genpath(pwd)); addpath(fileparts(mfilename('fullpath'))); define_constants;
o = mpoption('verbose',0,'out.all',0,'exp.use_legacy_core',1);
mpc0 = modifyMPC();
stamps = datetime(2019,1,1,0,0,0) + hours(0:8759);
variants = {'full','noJ','noL','noE','projOnly'};
fid = fopen(outfile,'w');
fprintf(fid,'ts,nyload,variant,ok,redisp_ny,ny_gen,redisp_ext,ny_vmin,ny_vmax\n');
for k = worker:nworkers:numel(stamps)
  ts = stamps(k);
  try
    [~,mf] = updateOpCond(mpc0,ts,false,false,true);
    nyb = mf.bus(ismember(mf.bus(:,BUS_AREA),[2 3]),BUS_I); ny = ismember(mf.bus(:,BUS_I),nyb);
    for v = 1:numel(variants)
      x = build(mf, mpc0, variants{v});
      [rp,~] = acproj(x,o);
      nyg = ismember(x.gen(:,GEN_BUS),nyb);
      if rp.success
        dp = rp.gen(:,PG) - x.gen(:,PG);
        vals = [sum(abs(dp(nyg)))/2, sum(x.gen(nyg,PG)), sum(abs(dp(~nyg)))/2, min(rp.bus(ny,VM)), max(rp.bus(ny,VM))];
      else
        vals = [nan, sum(x.gen(nyg,PG)), nan(1,3)];
      end
      fprintf(fid,'%s,%.1f,%s,%d%s\n', datestr(ts,'yyyy-mm-dd HH'), sum(mf.bus(ny,PD)), variants{v}, rp.success, sprintf(',%.4f',vals));
    end
    if mod(k,100) == worker, fprintf('%s done\n', datestr(ts,'yyyy-mm-dd HH')); end
  catch e
    fprintf(fid,'%s,ERR,%s\n', datestr(ts,'yyyy-mm-dd HH'), strrep(e.message,',',';'));
  end
end
fclose(fid);
end

function x = build(mf, mpc0, variant)
% The hour's case under one variant of the recipe.
switch variant
  case 'full',     x = extbase(fixes(mf,'JL'), mpc0);
  case 'noJ',      x = extbase(fixes(mf,'L'), mpc0);
  case 'noL',      x = extbase(fixes(mf,'J'), mpc0);
  case 'noE',      x = fixes(mf,'JL');
  case 'projOnly', x = mf;
end
end
