// GitHub requests go directly from the user's browser to api.github.com.
// Credentials never enter plan exports, URLs, logs, or browser storage.
export async function launchQuantization({repo,branch,planPath,token,plan,request=fetch,onStatus=()=>{}}){
 if(!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo))throw Error('Enter a valid owner/repository.');
 if(!branch||/[\r\n]/.test(branch))throw Error('Enter a branch.');
 if(!/^[A-Za-z0-9_./-]+\.json$/.test(planPath)||planPath.startsWith('/')||planPath.split('/').some(p=>p==='..'||p===''))throw Error('Use a repository-relative JSON plan path.');
 if(!token.trim())throw Error('Enter a GitHub fine-grained token in the password field. It needs Contents and Actions read/write for this repository.');
 if(!plan?.quant_job||!Array.isArray(plan.precision))throw Error('Missing quantization plan.');
 const api=`https://api.github.com/repos/${repo}`,headers={Accept:'application/vnd.github+json',Authorization:`Bearer ${token.trim()}`,'Content-Type':'application/json','X-GitHub-Api-Version':'2026-03-10'};
 async function call(path,method='GET',body){
  let response;try{response=await request(api+path,{method,headers,redirect:'error',...(body?{body:JSON.stringify(body)}:{})});}catch{throw Error(method==='POST'?'Dispatch response was lost. Check GitHub Actions before retrying; a run may already exist.':'Could not reach GitHub. Check your network and repository access.');}
  if(!response.ok){const hint=response.status===401?'Token was not accepted.':response.status===403?'Check token permissions and repository Actions policy.':response.status===404?'Repository, branch, or workflow was not found—or the token cannot access it.':response.status===422?'Check the workflow inputs and branch.':'';throw Error(`GitHub ${method} failed (${response.status}). ${hint}`);}
  return response.status===204?null:response.json();
 }
 onStatus('Checking workflow access…');
 const wf=await call('/actions/workflows/atlas-quant.yml');if(wf.state!=='active')throw Error('The Atlas workflow is disabled in GitHub.');
 await call(`/contents/.github/workflows/atlas-quant.yml?ref=${encodeURIComponent(branch)}`);
 const id=globalThis.crypto.randomUUID(),path=planPath.replace(/\.json$/,`-${id}.json`);
 onStatus('Saving a new plan snapshot to your repository…');
 const bytes=new TextEncoder().encode(JSON.stringify(plan,null,2));let binary='';for(const b of bytes)binary+=String.fromCharCode(b);
 const saved=await call('/contents/'+path.split('/').map(encodeURIComponent).join('/'),'PUT',{message:'Save Atlas quantization plan snapshot',branch,content:btoa(binary)});
 if(!saved.commit?.sha)throw Error('GitHub did not return the plan commit. No run was submitted.');
 onStatus('Plan saved. Submitting the workflow…');
 const run=await call('/actions/workflows/atlas-quant.yml/dispatches','POST',{ref:branch,inputs:{plan_path:path,plan_commit:saved.commit.sha}});
 return {commit:saved.commit.sha,planPath:path,url:run?.html_url||`https://github.com/${repo}/actions/workflows/atlas-quant.yml`,runId:run?.workflow_run_id??null};
}
