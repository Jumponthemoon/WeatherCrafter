import os
import requests
import base64
import time
import json
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import numpy as np

class KontextBatchProcessor:
    def __init__(self, api_key: str, max_workers: int = 2):
        """
        Initialize batch processor for Kontext API
        
        Args:
            api_key: Your BFL API key
            max_workers: Maximum concurrent requests (6 for kontext-max, 24 for kontext-pro)
        """
        self.api_key = api_key
        self.max_workers = max_workers
        self.base_url =  "https://api.bfl.ai/v1/flux-kontext-pro"
        self.headers = {
            'accept': 'application/json',
            'x-key': api_key,
            'Content-Type': 'application/json',
        }


    def resize_to_2_1(self, img: Image.Image, keep="width"):
        w, h = img.size
        new_w = 4096
        new_h = 4096
        return img.resize((new_w, new_h), Image.LANCZOS)
    
    def encode_image(self, image_path: str) -> Tuple[str, Tuple[int, int]]:
        """Convert image to base64 string and return original size"""
        try:
            image = Image.open(image_path)
            original_size = image.size  # store original W×H
            image = self.resize_to_2_1(image)
            buffered = BytesIO()
            
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
            
            return base64.b64encode(buffered.getvalue()).decode(), original_size
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return None, None

    
    def submit_single_request(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        img_str, original_size = self.encode_image(image_path)
        if not img_str:
            return {"error": f"Failed to encode {image_path}"}
        
        payload = {
            'prompt': prompt,
            'input_image': img_str,
            **kwargs
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return {
                'request_id': data['id'],
                'polling_url': data.get('polling_url', f"{self.base_url.replace('/flux-kontext-pro', '/get_result')}"),
                'original_path': image_path,
                'prompt': prompt,
                'status': 'submitted',
                'original_size': original_size  # <-- keep original size
            }
        except requests.exceptions.RequestException as e:
            return {
                'error': str(e),
                'original_path': image_path,
                'prompt': prompt,
                'status': 'failed'
            }


    def poll_single_result(self, task: Dict[str, Any], 
                        max_wait_seconds: int = 300, 
                        poll_interval: float = 0.5) -> Dict[str, Any]:
        if 'error' in task:
            return task

        request_id = task['request_id']
        polling_url = task['polling_url']

        start = time.time()
        while True:
            # hard stop if it takes too long
            if time.time() - start > max_wait_seconds:
                task.update({'status': 'failed', 'error': f"Timeout after {max_wait_seconds}s"})
                return task
            try:
                time.sleep(poll_interval)
                resp = requests.get(
                    polling_url,
                    headers={'accept': 'application/json', 'x-key': self.api_key},
                    params={'id': request_id},
                    timeout=20  # <-- critical: avoid hanging forever
                )
                resp.raise_for_status()
                result = resp.json()

                status = str(result.get('status', '')).lower()
                # treat common “done” variants as ready
                if status in ('ready', 'succeeded', 'completed', 'done'):
                    sample = result.get('result', {}).get('sample')
                    if not sample:
                        task.update({'status': 'failed', 'error': 'Missing result sample URL'})
                        return task
                    task.update({'status': 'ready', 'result_url': sample, 'result': result})
                    return task
                elif status in ('error', 'failed'):
                    task.update({'status': 'failed', 'error': result.get('error', 'Generation failed')})
                    return task
                # otherwise loop again
            except requests.exceptions.RequestException as e:
                # transient network errors: you can either fail fast or continue with backoff
                task.update({'status': 'failed', 'error': f"Polling error: {e}"})
                return task


    def download_image(self, task: Dict[str, Any], output_dir: str = "./edited_images", target_weather: str = "unknown") -> Dict[str, Any]:
        if task['status'] != 'ready' or 'result_url' not in task:
            return task

        try:
            os.makedirs(output_dir, exist_ok=True)

            # Keep original name + extension
            original_name, original_ext = os.path.splitext(os.path.basename(task['original_path']))
            timestamp = int(time.time())
            filename = f"{original_name}_{target_weather}{original_ext.lower()}"   # <-- preserve extension
            filepath = os.path.join(output_dir, filename)

            # Download image
            img_response = requests.get(task['result_url'], timeout=50)
            img_response.raise_for_status()
            image = Image.open(BytesIO(img_response.content))
            print(image.size)  # (width, height)
            # Resize back to original dimensions if available
            if 'original_size' in task and task['original_size']:
                image = image.resize(task['original_size'], Image.LANCZOS)

            # Save using original extension (convert to RGB if needed)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            # Map extension to valid Pillow format
            ext_to_format = {
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".png": "PNG",
                ".bmp": "BMP",
            }
            fmt = image.format if image.format else ext_to_format.get(original_ext.lower(), "JPEG")

            image.save(filepath, format=fmt)

            task.update({
                'local_path': filepath,
                'downloaded': True
            })
            print(f"Downloaded (resized to {task['original_size']}): {filepath}")

        except Exception as e:
            task.update({
                'download_error': str(e),
                'downloaded': False
            })
            print(f"Download failed for {task['original_path']}: {e}")

        return task

    def process_batch(self, 
                     image_prompts: List[Tuple[str, str]], 
                     output_dir: str = "./edited_images",
                     target_weather: str = "unknown",
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            image_prompts: List of (image_path, prompt) tuples
            output_dir: Directory to save edited images
            **kwargs: Additional parameters for the API
            
        Returns:
            List of task results with download information
        """
        print(f"Starting batch processing of {len(image_prompts)} images...")
        
        # Submit all requests
        print("Submitting requests...")
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.submit_single_request, img_path, prompt, **kwargs): (img_path, prompt)
                for img_path, prompt in image_prompts
            }
            
            for future in as_completed(future_to_task):
                task = future.result()
                tasks.append(task)
                if 'error' not in task:
                    print(f"Submitted: {task['original_path']}")
                else:
                    print(f"Failed to submit: {task['original_path']} - {task['error']}")
        
        # Poll for results
        print("Polling for results...")
        completed_tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.poll_single_result, task): task
                for task in tasks if 'error' not in task
            }
            
            for future in as_completed(future_to_task):
                completed_task = future.result()
                completed_tasks.append(completed_task)
                if completed_task['status'] == 'ready':
                    print(f"Ready: {completed_task['original_path']}")
                else:
                    print(f"Failed: {completed_task['original_path']}")
        
        # Add failed submissions to completed tasks
        completed_tasks.extend([task for task in tasks if 'error' in task])
        
        # Download all successful results
        print("Downloading images...")
        final_results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {
                executor.submit(self.download_image, task, output_dir,target_weather): task
                for task in completed_tasks
            }
            
            for future in as_completed(future_to_task):
                final_result = future.result()
                final_results.append(final_result)
        
        # Print summary
        successful = len([r for r in final_results if r.get('downloaded', False)])
        failed = len(final_results) - successful
        print(f"\nBatch processing complete!")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        
        return final_results
    
    def retry_failed_tasks(self, failed_tasks: List[Dict[str, Any]], output_dir: str,
                        max_retries: int = 2, retry_delay: int = 10,
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Retry failed image generation tasks.

        Args:
            failed_tasks: List of failed task dicts from a previous run
            output_dir: Directory to save retried results
            max_retries: How many times to retry each failed task
            retry_delay: Seconds to wait between retries
            **kwargs: Extra API parameters (prompt_upsampling, seed, etc.)

        Returns:
            List of updated task results (successful or failed)
        """
        retried_results = []

        for attempt in range(1, max_retries + 1):
            still_failed = [t for t in failed_tasks if t.get("status") != "ready"]

            if not still_failed:
                break

            print(f"\nRetry attempt {attempt}/{max_retries} for {len(still_failed)} failed tasks...")
            time.sleep(retry_delay)  # Backoff between rounds

            # Re-submit
            new_requests = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self.submit_single_request, t['original_path'], t['prompt'], **kwargs): t
                    for t in still_failed
                }
                for future in as_completed(future_to_task):
                    result = future.result()
                    new_requests.append(result)

            # Poll and download
            retried_batch = self.process_batch(
                [(t['original_path'], t['prompt']) for t in new_requests if 'error' not in t],
                output_dir=output_dir,
                **kwargs
            )

            # Merge results
            for r in retried_batch:
                # If it succeeded, replace the original failed record
                for orig in failed_tasks:
                    if orig['original_path'] == r['original_path']:
                        orig.update(r)
            retried_results.extend(retried_batch)

        return failed_tasks

def chunk_list(lst, n):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_edit(image_prompts, output_dir, target_weather, error_handling=False,
             api_key=None, max_workers=6, batch_size=20, model_seed=2):
    """Submit, poll, download (and optionally retry) a list of (image_path, prompt) edits.

    Returns the list of per-task result dicts. The BFL API key is read from the
    BFL_API_KEY environment variable when not passed explicitly.
    """
    api_key = api_key or os.environ.get("BFL_API_KEY")
    if not api_key:
        print("Please set BFL_API_KEY environment variable")
        return []

    processor = KontextBatchProcessor(api_key, max_workers=max_workers)

    all_results = []
    for batch_idx, batch in enumerate(chunk_list(image_prompts, batch_size), start=1):
        print(f"Processing batch {batch_idx} with {len(batch)} images...")
        results = processor.process_batch(
            batch,
            target_weather=target_weather,
            output_dir=output_dir,
            prompt_upsampling=False,
            seed=model_seed,
            safety_tolerance=0,
        )
        all_results.extend(results)

        # Save intermediate results (in case the run stops midway).
        with open("batch_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Finished batch {batch_idx}, results appended to batch_results.json")

    print(f"All {len(all_results)} images processed.")

    if error_handling:
        failed_tasks = [r for r in all_results if not r.get('downloaded', False)]
        if failed_tasks:
            print(f"\n{len(failed_tasks)} tasks failed. Retrying...")
            retried = processor.retry_failed_tasks(
                failed_tasks,
                output_dir=output_dir,
                max_retries=10,
                retry_delay=15,
                prompt_upsampling=False,
                seed=model_seed,
                safety_tolerance=0,
            )
            for r in retried:
                for i, orig in enumerate(all_results):
                    if orig['original_path'] == r['original_path']:
                        all_results[i] = r
            print(f"Retry phase complete. Total successful: "
                  f"{len([r for r in all_results if r.get('downloaded', False)])}")

    return all_results


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input_dir", "-i", default="./input_images", help="Input folder with images.")
    parser.add_argument("--output_dir", "-o", default="./edited_images", help="Folder to save results.")
    parser.add_argument("--style", "-s", default="snowy", help="Prompt style to apply.")
    parser.add_argument("--target_weather", default="snowy", help="Target weather condition to apply.")
    parser.add_argument("--error_handling", action="store_true")
    parser.add_argument("--single_image", "-si", default=None, help="Single image path (overrides input_dir).")


def main(args: argparse.Namespace) -> None:
    allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    if args.single_image:
        if not os.path.exists(args.single_image):
            print(f"Image not found: {args.single_image}")
            return
        image_prompts = [(args.single_image, args.style)]
        print(f"Single image mode: {args.single_image}")
    else:
        image_prompts = [
            (os.path.join(args.input_dir, filename), args.style)
            for filename in os.listdir(args.input_dir)
            if filename.lower().endswith(allowed_extensions)
        ]
        print(f"Batch mode: {len(image_prompts)} images from {args.input_dir}")

    run_edit(image_prompts, args.output_dir, args.target_weather, error_handling=args.error_handling)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Batch process images with weather/style prompts.")
    add_arguments(_parser)
    main(_parser.parse_args())
