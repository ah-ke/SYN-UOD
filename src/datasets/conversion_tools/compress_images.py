from pathlib import Path
import click
import asyncio, sys
from tqdm import tqdm
import logging

log = logging.getLogger('compress_images')
log.setLevel(logging.DEBUG)

def asyncio_loop():
    if sys.platform == 'win32':
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        return loop
    return asyncio.get_event_loop()

def async_map(func, tasks, num_concurrent=4):
	num_tasks = len(tasks)
	queue = asyncio.Queue()

	for idx, task in enumerate(tasks):
		queue.put_nowait((idx, task))

	results = [None] * num_tasks
	pbar = tqdm(total=num_tasks)

	async def async_worker():
		while not queue.empty():
			idx, task = await queue.get()
			result = await func(task)
			results[idx] = result
			pbar.update(1)
			queue.task_done()

	async def run_tasks():
		workers = [async_worker() for _ in range(num_concurrent)]
		await asyncio.gather(*workers)

	loop = asyncio_loop()
	loop.run_until_complete(run_tasks())
	pbar.close()

	return results

async def run_external_program(cmd):

	try:
		proc = await asyncio.create_subprocess_exec(
			*map(str, cmd), # str to convert Path objects
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, stderr = await proc.communicate()
	except Exception as e:
		cmd_as_str = ' '.join(map(str, cmd))
		log.exception(f'run_externam_program({cmd_as_str})')
		raise e

	if proc.returncode != 0:
		cmd_as_str = ' '.join(map(str, cmd))
		log.error(f"""Command {cmd_as_str} error, retcode = {proc.returncode}
--Stderr--
{stderr}
--Stdout--
{stdout}""")


# def cmd_jpg(src, dest):
# 	return ['convert', src, '-quality', '80', dest.with_suffix('.jpg')]

# def cmd_webp(src, dest):
# 	return ['cwebp',  
# 		src, '-o', dest.with_suffix('.webp'), 
# 		'-q', '90',
# 		'-sharp_yuv', 
# 		'-m', '6',
# 	]

# "{src} -o {dest} -q 90 -sharp_yuv -m 6"


@click.command()
# Cityscapes
# @click.argument('src_dir', type=click.Path(exists=True, dir_okay=True), default="G:/Dataset/cityscapes/leftImg8bit")
# @click.argument('dest_dir', type=click.Path(), default='G:/Anomaly/Detecting/detecting-the-unexpected-master/datasets/dataset_Cityscapes/1024x512/leftImg8bit')
# LostAndFound
# @click.argument('src_dir', type=click.Path(exists=True, dir_okay=True), default="G:/Dataset/LostAndFound/leftImg8bit")
# @click.argument('dest_dir', type=click.Path(), default='G:/Anomaly/Detecting/detecting-the-unexpected-master/datasets/dataset_LostAndFound/1024x512/leftImg8bit')
# @click.argument('cmd', type=str, default="cwebp {src} -o {dest} -q 90 -sharp_yuv -m 6 -resize 1024 512")
# @click.option('--ext', default='.webp')
# @click.option('--concurrent', default=20)
# Fishyscapes
# @click.argument('src_dir', type=click.Path(exists=True, dir_okay=True), default="G:/Dataset/Fishyscapes/images")
# @click.argument('dest_dir', type=click.Path(), default='G:/Anomaly/Detecting/Detecting_by_resynthesis/datasets/dataset_Fishyscapes/1024x512/images')
# @click.argument('cmd', type=str, default="cwebp {src} -o {dest} -q 90 -sharp_yuv -m 6 -resize 1024 512")
# @click.option('--ext', default='.png')
# @click.option('--concurrent', default=20)

# # Cityscapes
# # @click.argument('src_dir', type=click.Path(exists=True, dir_okay=True), default="G:/Dataset/cityscapes/gtFine")
# # @click.argument('dest_dir', type=click.Path(), default='G:/Anomaly/Detecting/detecting-the-unexpected-master/datasets/dataset_Cityscapes/1024x512_webp/gtFine')
# # LostAndFound
# @click.argument('src_dir', type=click.Path(exists=True, dir_okay=True), default="G:/Dataset/LostAndFound/gtCoarse")
# @click.argument('dest_dir', type=click.Path(), default='G:/Anomaly/Detecting/detecting-the-unexpected-master/datasets/dataset_LostAndFound/1024x512/gtCoarse')
# @click.argument('cmd', type=str, default="magick {src} -filter point -resize 50% {dest}")
# @click.option('--ext', default='.png')
# @click.option('--concurrent', default=20)
# Fishyscapes
@click.argument('src_dir', type=click.Path(exists=True, dir_okay=True), default="G:/Dataset/Fishyscapes/labels_masks")
@click.argument('dest_dir', type=click.Path(), default='G:/Anomaly/Detecting/Detecting_by_resynthesis/datasets/dataset_Fishyscapes/1024x512/labels_masks')
@click.argument('cmd', type=str, default="magick {src} -filter point -resize 50% {dest}")
@click.option('--ext', default='.png')
@click.option('--concurrent', default=20)

def main(src_dir, dest_dir, cmd, ext, concurrent):
	src_dir = Path(src_dir)
	dest_dir = Path(dest_dir)

	log.info('Collecting files')
	src_files = list(src_dir.glob('**/*.png'))
	dest_files = [dest_dir / p.relative_to(src_dir) for p in src_files]

	dirs_to_create = set(p.parent for p in dest_files)
	log.info(f'Creating {dirs_to_create.__len__()} dirs')
	for par in dirs_to_create:
		par.mkdir(parents=True, exist_ok=True)
	
	cmds = [
		cmd.format(src=src_file, dest=dest_file.with_suffix(ext)).split() 
		for (src_file, dest_file) in zip(src_files, dest_files)
	]

	async_map(run_external_program, cmds, num_concurrent=concurrent)

if __name__ == '__main__':
	main()
