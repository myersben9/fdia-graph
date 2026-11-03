function ac_feas(mpdir, hours, outfile)
% AC feasibility of NYgrid: for each hour, rebuild NYgrid's operating point, then
% run their DC PF (reference), AC PF on the reduced 57-bus case, AC PF with Q limits,
% and AC PF on the full modified NPCC-140 case.
addpath(genpath(mpdir)); addpath(genpath(pwd));
define_constants;
mpc0 = modifyMPC();
o  = mpoption('verbose',0,'out.all',0,'exp.use_legacy_core',1);
oq = mpoption(o,'pf.enforce_q_lims',1);
fid = fopen(outfile,'w');
fprintf(fid,'ts,dc_ok,acr_ok,acr_it,acr_vmin,acr_vmax,acr_maxload,acr_loss,acr_slackP,acr_slackQ,acrq_ok,acrq_vmin,acrq_vmax,acf_ok,acf_vmin,acf_vmax,acf_ny_vmin,acf_ny_vmax,acf_maxload,acf_loss,acf_slackP,dc_ifmae,ac_ifmae,acdc_ifmaxdiff,nyload\n');
for k = 1:numel(hours)
  ts = hours(k);
  try
    [mr,mf] = updateOpCond(mpc0,ts,false,false,true);
    [~,interFlow,flowLimit,~,~,~] = readOpCond(ts);
    dc = rundcpf(mr,o); acr = runpf(mr,o); acq = runpf(mr,oq); acf = runpf(mf,o);
    ny = acf.bus(:,BUS_I)>=37 & acf.bus(:,BUS_I)<=82;
    [~,~,edc] = flow4Plot(dc,interFlow,flowLimit);
    if acr.success, [fac,~,eac] = flow4Plot(acr,interFlow,flowLimit); [fdc] = flow4Plot(dc,interFlow,flowLimit); d = max(abs(fac-fdc)); m2 = mean(abs(eac)); else, d = NaN; m2 = NaN; end
    fprintf(fid,'%s,%d,%d,%d,%.4f,%.4f,%.3f,%.2f,%.1f,%.1f,%d,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.2f,%.1f,%.4f,%.4f,%.1f,%.1f\n', ...
      datestr(ts,'yyyy-mm-dd HH'), dc.success, acr.success, acr.iterations, min(acr.bus(:,VM)), max(acr.bus(:,VM)), ml(acr), sum(real(get_losses(acr))), slk(acr,PG), slk(acr,QG), ...
      acq.success, min(acq.bus(:,VM)), max(acq.bus(:,VM)), acf.success, min(acf.bus(:,VM)), max(acf.bus(:,VM)), min(acf.bus(ny,VM)), max(acf.bus(ny,VM)), ml(acf), sum(real(get_losses(acf))), slk(acf,PG), ...
      mean(abs(edc)), m2, d, sum(mf.bus(ny,PD)));
    fprintf('%s dc=%d acr=%d(it %d, V %.3f-%.3f) acq=%d acf=%d(V %.3f-%.3f) ifMAE dc=%.3f ac=%.3f\n', datestr(ts,'yyyy-mm-dd HH'), dc.success, acr.success, acr.iterations, min(acr.bus(:,VM)), max(acr.bus(:,VM)), acq.success, acf.success, min(acf.bus(:,VM)), max(acf.bus(:,VM)), mean(abs(edc)), m2);
  catch e
    fprintf(fid,'%s,ERR %s\n', datestr(ts,'yyyy-mm-dd HH'), strrep(e.message,',',';'));
    fprintf('%s ERR %s\n', datestr(ts,'yyyy-mm-dd HH'), getReport(e,'extended','hyperlinks','off'));
  end
end
fclose(fid);
end
function v = ml(r)
  define_constants; s = max(abs(r.branch(:,PF)+1j*r.branch(:,QF)), abs(r.branch(:,PT)+1j*r.branch(:,QT)));
  ra = r.branch(:,RATE_A); k = ra>0; if any(k), v = max(s(k)./ra(k)); else, v = NaN; end
end
function v = slk(r,col)
  define_constants; ref = r.bus(r.bus(:,BUS_TYPE)==REF,BUS_I); v = sum(r.gen(ismember(r.gen(:,GEN_BUS),ref),col));
end
