"""
Parallel Processor for FactoryMind AI Ingestion Pipeline.
Processes pages in parallel using ThreadPoolExecutor.
Shows progress bar and handles errors gracefully.
"""
import logging
from typing import List, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger("factorymind")


class ParallelProcessor:
    """
    Processes tasks in parallel with progress tracking.
    Never crashes - handles individual task failures gracefully.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.stats = {
            "total_tasks": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
    
    def process_pages(self, pages: List[int], process_func: Callable, 
                     desc: str = "Processing") -> Dict[int, Any]:
        """
        Process pages in parallel.
        
        Args:
            pages: List of page numbers to process
            process_func: Function to call for each page (receives page_num)
            desc: Description for progress bar
        
        Returns:
            Dictionary mapping page_num to result
        """
        self.stats["total_tasks"] = len(pages)
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(process_func, page_num): page_num
                for page_num in pages
            }
            
            # Process with progress bar
            with tqdm(total=len(pages), desc=desc, unit="page") as pbar:
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    
                    try:
                        result = future.result()
                        results[page_num] = result
                        self.stats["successful"] += 1
                        pbar.set_postfix({"success": self.stats["successful"], "failed": self.stats["failed"]})
                    except Exception as e:
                        self.stats["failed"] += 1
                        error_msg = f"Page {page_num}: {str(e)}"
                        self.stats["errors"].append(error_msg)
                        logger.warning(error_msg)
                        results[page_num] = None
                    
                    pbar.update(1)
        
        return results
    
    def process_with_retry(self, tasks: List[Any], process_func: Callable,
                          max_retries: int = 2, desc: str = "Processing") -> Dict[Any, Any]:
        """
        Process tasks with automatic retry on failure.
        
        Args:
            tasks: List of tasks to process
            process_func: Function to call for each task
            max_retries: Maximum retry attempts
            desc: Description for progress bar
        
        Returns:
            Dictionary mapping task to result
        """
        self.stats["total_tasks"] = len(tasks)
        results = {}
        
        def process_with_retries(task):
            """Process a single task with retries."""
            for attempt in range(max_retries + 1):
                try:
                    return process_func(task)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    logger.debug(f"Retry {attempt + 1}/{max_retries} for task {task}")
            
            return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(process_with_retries, task): task
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc=desc, unit="task") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        results[task] = result
                        self.stats["successful"] += 1
                        pbar.set_postfix({"success": self.stats["successful"], "failed": self.stats["failed"]})
                    except Exception as e:
                        self.stats["failed"] += 1
                        error_msg = f"Task {task}: {str(e)}"
                        self.stats["errors"].append(error_msg)
                        logger.warning(error_msg)
                        results[task] = None
                    
                    pbar.update(1)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*70)
        print("  PARALLEL PROCESSING STATISTICS")
        print("="*70)
        print(f"  Total tasks:        {self.stats['total_tasks']}")
        print(f"  Successful:         {self.stats['successful']}")
        print(f"  Failed:             {self.stats['failed']}")
        
        if self.stats['errors']:
            print(f"\n  Errors:")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")
            if len(self.stats['errors']) > 5:
                print(f"    ... and {len(self.stats['errors']) - 5} more errors")
        
        print("="*70 + "\n")
