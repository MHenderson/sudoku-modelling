env = Environment()
local_texmf = '/home/matthew/texmf//'
local_texmf += ':.:'
env['TEXINPUTS'] = local_texmf
env['BIBINPUTS'] = local_texmf
env['PDFLATEXCOM'] = 'cd ${TARGET.dir} && export TEXINPUTS=$TEXINPUTS && $PDFLATEX $PDFLATEXFLAGS ${SOURCE.file}'
env['BIBTEXCOM'] = 'cd ${TARGET.dir} && export BIBINPUTS=$BIBINPUTS && $BIBTEX $BIBTEXFLAGS ${SOURCE.file}'
Export('env')
SConscript(['SConscript'],build_dir='/home/matthew/Documents/scipy2010')
