# Java 使用阻塞队列 BlockingQueue 多线程在一个目录及它的所以子目录下搜索所有文件，打印出包含关键字的行

#### 阻塞队列（ blocking queue )

生产者线程向队列插人元素， 消费者线程则取出它们。使用队列，可以安全地从一个线程向另一个线程传递数据。 工作者线程可以周期性地将中间结果存储在阻塞队列中。其他的工作者线程移出中间结果并进一步加以修改。队列会自动地平衡负载。如果第一个线程集运行得比第二个慢， 第二个线程集在等待结果时会阻塞。 如果第一个线程集运行得快， 它将等待第二个队列集赶上来。

**java.util.concurrent** 包提供了阻塞队列的几个变种。
- **LinkedBlockingQueue** 的容量是没有上边界的但是，也可以选择指定最大容量。**LinkedBlockingDeque** 是一个双端的版本。
- **ArrayBlockingQueue** 在构造时需要指定容量，并且有一个可选的参数来指定是否需要公平性。若设置了公平参数， 则那么等待了最长时间的线程会优先得到处理。通常，公平性会降低性能，只有在确实非常需要时才使用它。
- **PriorityBlockingQueue** 是一个带优先级的队列， 而不是先进先出队列。元素按照它们的优先级顺序被移出。该队列是没有容量上限，但是 ，如果队列是空的， 取元素的操作会阻塞。
> 常用的方法有两个，put方法添加元素，take方法取出元素。如果队列满， 则 put 方法阻塞 ； 如果队列空， 则 take 方法阻塞。

以下程序展示了如何使用阻塞队列来控制一组线程。程序在一个目录及它的所有子目录下搜索所有文件， 打印出包含指定关键字的行

```Java
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * 使用BlockingQueue多线程技术，在一个目录及它的所以子目录下搜索所有文件，打印出包含关键字的文件及行号
 *
 */
public class BlcokingQueueTest {

	private static final int FILE_QUEUE_SIZE = 10;
	private static final int SEARCH_THREADS = 100; // 搜索线程数
	private static final File DUMMY = new File(""); // 虚拟空文件夹，作为线程结束标志
	// 阻塞队列，先进先出
	private static BlockingQueue<File> queue = new ArrayBlockingQueue<>(FILE_QUEUE_SIZE);

	public static void main(String[] args) {

		try (Scanner in = new Scanner(System.in)) {
			System.out.print("Enter base directory (e.g. /opt/jdk1.8.0/src): ");
			String directory = in.nextLine();
			System.out.print("Enter keyword (e.g. BlockingQueue): ");
			String keyword = in.nextLine();

			// 开启一个线程，递归枚举给定目录及其所有子目录下所有文件的文件名
			Runnable enumerator = () -> {
				try {
					enumerate(new File(directory));
					queue.put(DUMMY);  // 最后添加进空文件，以空文件作为结束标志
				} catch (InterruptedException e) {
				}
			};
			new Thread(enumerator).start();

			// 开启 SEARCH_THREADS(100) 个线程，查找所有文件中具有keyword的行
			for (int i = 1; i <= SEARCH_THREADS; i++) {
				Runnable searcher = () -> {
					try {
						boolean done = false;
						while (!done) {
							File file = queue.take();  // 取出队列中的文件名
							if (file == DUMMY) {
								queue.put(file);   // 空文件则说明已经结束
								done = true;
							} else
								search(file, keyword);
						}
					} catch (IOException e) {
						e.printStackTrace();
					} catch (InterruptedException e) {
					}
				};
				new Thread(searcher).start();
			}
		}
	}

	/**
	 * 递归枚举给定目录及其所有子目录下所有文件的文件名
	 * 
	 * @param directory
	 *            开始搜索的目录
	 * @throws InterruptedException
	 */
	public static void enumerate(File directory) throws InterruptedException {
		File[] files = directory.listFiles(); // 返回目录下所有文件名和目录的list迭代器
		for (File file : files) {
			if (file.isDirectory()) // 判断是否的目录
				enumerate(file); // 对于目录继续递归枚举
			else
				queue.put(file); // 文件则将文件名加入队列
		}
	}

	/**
	 * 打开文件，搜索文件中包含keyword的行
	 * 
	 * @param file
	 *            搜索关键字的文件
	 * @param keyword
	 *            用于搜索的关键字
	 * @throws IOException
	 */
	public static void search(File file, String keyword) throws IOException {
		try (Scanner in = new Scanner(file, "UTF-8")) {
			int lineNumber = 0;
			while (in.hasNextLine()) {
				lineNumber ++;
				String line = in.nextLine();
				if (line.contains(keyword))    // 打印文件中包含 keyword 的行的信息
					System.out.printf("%s:%d:%s%n", file.getPath(), lineNumber, line);
			}
		}
	}
}

```

> 以上代码来自 Java 核心技术卷一第14章。