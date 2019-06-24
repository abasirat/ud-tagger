import sys

def inc_dict_value(d,k): 
  try: 
    d[k] += 1  
  except KeyError: 
    d[k] = 1

def update_progress(progress, prefix="Precent", barLength=10) :
  # copied from https://stackoverflow.com/questions/3160699/python-progress-bar
  status = ""
  if isinstance(progress, int): progress = float(progress)
  if not isinstance(progress, float): 
    progress = 0
    status = "error: progress var must be float\n"
  if progress < 0: 
    progress = 0
    status = "Halt...\n"
  if progress >= 1: 
    progress = 1
    status = "Done...\n"
  block = int(round(barLength*progress))
  text = "\r{0}: [{1}] {2:.2f}% {3}".format( prefix , "#"*block + "-"*(barLength-block), progress*100, status)
  sys.stdout.write(text)
  sys.stdout.flush()

def match(file1,file2,outfile) :
  joint_dict = {}
  t1_dict = {}
  t2_dict = {}
  t1_and_t2 = {}

  with open(file1,"r") as f1, open(file2,"r") as f2 :
    for line1 , line2 in zip(f1,f2) :
      tok1 = line1.rstrip().split() 
      tok2 = line2.rstrip().split() 
      matched = [t1 + '_' + t2 for t1,t2 in zip(tok1,tok2)]
      for x in matched: inc_dict_value(joint_dict,x)
      for x in tok1: inc_dict_value(t1_dict,x)
      for x in tok2: inc_dict_value(t2_dict,x)
  T = 0 
  for key in list(t1_dict.keys()) : T += t1_dict[key] 
  for key in list(joint_dict.keys()) :
    (t1,t2) = key.split('_')
    t1_and_t2[key] = joint_dict[key]*1.0/(t1_dict[t1])
  with open(outfile,"w") as outfp:
    for key in sorted(t1_and_t2, key=t1_and_t2.get, reverse=True) :
      (t1,t2) = key.split('_')
      print("{0} {1} {2:1.3f} {3:1.3f} {4:1.3f} ".format(t1,t2,t1_dict[t1]*1.0/T,t2_dict[t2]*1.0/T,t1_and_t2[key]),file=outfp)
  return


