function dump_hours(mpdir, outdir)
addpath(genpath(mpdir)); addpath(genpath(pwd)); define_constants;
mpc0 = modifyMPC(); m1 = mpc0;
ts = [datetime(2019,1,1,0,0,0), datetime(2019,4,15,14,0,0), datetime(2019,7,20,17,0,0)];
for k = 1:numel(ts)
  [~, mf] = updateOpCond(mpc0, ts(k), false, false, true);
  x = extbase(fixes(mf,'JL'), m1);
  tag = datestr(ts(k),'yyyymmddHH');
  writematrix([x.gen(:,GEN_BUS) x.gen(:,PG) x.gen(:,PMIN) x.gen(:,PMAX)], fullfile(outdir, ['gen_' tag '.csv']));
  writematrix([x.bus(:,BUS_I) x.bus(:,PD) x.bus(:,QD) x.bus(:,BUS_TYPE)], fullfile(outdir, ['bus_' tag '.csv']));
  [r, info] = acproj(x, mpoption('verbose',0,'out.all',0,'exp.use_legacy_core',1));
  writematrix([r.bus(:,VM) r.bus(:,VA)], fullfile(outdir, ['vm_' tag '.csv']));
  writematrix([r.gen(:,PG) r.gen(:,QG)], fullfile(outdir, ['acgen_' tag '.csv']));
  fprintf('%s ok=%d redisp=%.1f\n', tag, info.ok, info.redisp);
end
end
