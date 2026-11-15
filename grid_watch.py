import os, json, time, datetime
os.chdir(r'C:\SZL\cram-study')
rec = {'started': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'events': []}
json.dump(rec, open('grid_watch.json', 'w'), indent=2)
for i in range(360):
    if os.path.exists('diag5.json'):
        try:
            d = json.load(open('diag5.json'))
        except Exception:
            time.sleep(5); continue
        rec['diag5'] = d
        if not d.get('ok'):
            rec['decision'] = 'diag5_crashed_no_launch'
        elif d['learned100_mfa2']['exact'] > 0:
            rec['decision'] = 'validated_launch_grid'
        else:
            rec['decision'] = 'learned_not_validated_no_launch'
        break
    time.sleep(10)
else:
    rec['decision'] = 'timeout_1h_no_diag5'
rec['events'].append({'t': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision': rec.get('decision')})
json.dump(rec, open('grid_watch.json', 'w'), indent=2)
if rec.get('decision') == 'validated_launch_grid':
    import subprocess
    subprocess.Popen(['python', r'C:\SZL\cram-study\train_launch.py'], cwd=r'C:\SZL\cram-study', creationflags=0x00000008)
    rec['grid_launched'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    json.dump(rec, open('grid_watch.json', 'w'), indent=2)
    print('LAUNCHED')
else:
    print(rec.get('decision'))
