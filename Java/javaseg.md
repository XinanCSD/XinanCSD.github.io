# java 中文文本分词

本文使用 classifier4J 以及 IKAnalyzer2012_u6 实现中文分词。可以增加自定义词库，词库保存为 "exdict.dic" 文件，一个词一行。

```Java
// MyTokenizer.java 文件

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import net.sf.classifier4J.ITokenizer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.wltea.analyzer.cfg.Configuration;
import org.wltea.analyzer.cfg.DefaultConfig;
import org.wltea.analyzer.dic.Dictionary;
import org.wltea.analyzer.lucene.IKTokenizer;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

/**
 * 中文分词器类
 * 
 * @author CSD
 *
 */
@SuppressWarnings("deprecation")
public class MyTokenizer implements ITokenizer {

	private static final Logger logger = LogManager.getLogger(MyTokenizer.class);

	private List<String> list;
	private String[] strArray;
	private static Collection<String> exwordc = new ArrayList<>();
	private static String exdict = "exdict.dic";

	// 加载新增词库
	static {

		try {
			File file = new File(exdict);
			FileInputStream fin = new FileInputStream(file);
			BufferedReader reader = new BufferedReader(new InputStreamReader(fin));
			String line = "";
			while ((line = reader.readLine()) != null) {
				exwordc.add(line.trim());
			}
			reader.close();
			logger.info("加载词典::" + exdict);
			// 增加词库
			Configuration cfg = DefaultConfig.getInstance();
			Dictionary dict = Dictionary.initial(cfg);
			dict.addWords(exwordc);
		} catch (IOException e) {
			logger.error(e + "------------------加载词典出错，请确认词典文件！------------------");
		}
	}

	/**
	 * 分词，返回分词数组
	 * 
	 * @param input
	 *            文本字符串
	 * @return String[]
	 */
	public String[] tokenize(String input) {
		list = new ArrayList<String>();

		IKTokenizer tokenizer = new IKTokenizer(new StringReader(input), true);
		try {
			while (tokenizer.incrementToken()) {
				TermAttribute termAtt = (TermAttribute) tokenizer.getAttribute(TermAttribute.class);
				String str = termAtt.term();
				list.add(str);
			}
		} catch (IOException e) {
			logger.error(e + "------------------分词出错------------------");
		}
		strArray = new String[list.size()];
		for (int i = 0; i < list.size(); i++) {
			strArray[i] = (String) list.get(i);
		}

		return strArray;
	}

}

```



```Java
// Segmentation.java 
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import net.sf.classifier4J.ITokenizer;

/**
 * 中文语料分词
 * 
 * @author CSD
 *
 */
public class Segmentation {

	private static final Logger logger = LogManager.getLogger(Segmentation.class);

	public static void main(String[] args) throws IOException {

		String path = "1.txt";
		File file = new File(path);
		FileInputStream fin = new FileInputStream(file);
		String input = getString(fin);

		logger.info("开始分词::" + path);
		ITokenizer tokenizer = new MyTokenizer();
		String[] words = tokenizer.tokenize(input);
		for (String word : words) {
			System.out.println(word);
		}

	}

	/**
	 * 从 inputStream 读取文本并转为一个字符串。
	 * 
	 * @param is
	 *            inputStream 输入流
	 * @return String 文本字符串
	 * @throws IOException
	 */
	public static String getString(InputStream is) throws IOException {

		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		String line = "";
		StringBuffer stringBuffer = new StringBuffer();
		while ((line = reader.readLine()) != null) {
			stringBuffer.append(line);
			stringBuffer.append(" ");
		}

		reader.close();

		return stringBuffer.toString().trim();
	}
}

```


程序需依赖 IKAnalyzer2012_u6.jar 以及添加 pom.xml 文件
```xml
<!-- https://mvnrepository.com/artifact/classifier4j/classifier4j -->
<dependency>
    <groupId>classifier4j</groupId>
    <artifactId>classifier4j</artifactId>
    <version>0.6</version>
</dependency>
<!-- https://mvnrepository.com/artifact/org.apache.lucene/lucene-analyzers -->
<dependency>
    <groupId>org.apache.lucene</groupId>
    <artifactId>lucene-analyzers</artifactId>
    <version>3.2.0</version>
</dependency>

<dependency>
    <groupId>org.slf4j</groupId>
    <artifactId>slf4j-log4j12</artifactId>
    <version>1.7.5</version>
</dependency>
```
