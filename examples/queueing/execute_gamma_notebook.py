"""Execute notebook cells with IPython in process when kernel sockets are blocked."""
import io
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import display, Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show(*args, **kwargs):
    for number in plt.get_fignums():
        stream=io.BytesIO()
        plt.figure(number).savefig(stream,format='png',dpi=90,bbox_inches='tight')
        display(Image(data=stream.getvalue()))
plt.show=show
shell=InteractiveShell.instance()
p='examples/queueing/gamma_baseline.ipynb'
n=nbformat.read(p,as_version=4)
count=0
for cell in n.cells:
    if cell.cell_type!='code': continue
    count+=1
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=True)
    if result.error_before_exec or result.error_in_exec:
        print(captured.stdout,captured.stderr)
        raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
    outputs=[]
    if captured.stdout: outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr: outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for out in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
    cell.execution_count=count
    cell.outputs=outputs
    print('Executed code cell',count,'rich outputs',len(captured.outputs))
n.metadata['execution']={'method':'IPython in-process sequential execution; matplotlib Agg rich display',
                         'reason':'Execution environment blocks local kernel sockets'}
nbformat.validate(n)
nbformat.write(n,p)
