import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class InvertedIndex {

    // Mapper阶段：分词、过滤停词、输出<单词, 文件名,1>
    public static class IndexMapper extends Mapper<Object, Text, Text, Text> {
        private final HashSet<String> stopWordSet = new HashSet<>();
        private final Text outputKey = new Text();
        private final Text outputValue = new Text();

        // 任务启动时加载分布式缓存中的停词表
        @Override
        protected void setup(Context context) throws IOException {
            Path[] cacheFiles = context.getLocalCacheFiles();
            if (cacheFiles != null && cacheFiles.length > 0) {
                BufferedReader br = new BufferedReader(new FileReader(cacheFiles[0].toString()));
                String word;
                while ((word = br.readLine()) != null) {
                    stopWordSet.add(word.trim().toLowerCase());
                }
                br.close();
            }
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // 获取当前处理的文件名
            String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
            // 按非字母分割分词，统一转小写
            String[] words = value.toString().toLowerCase().split("[^a-zA-Z]+");

            for (String word : words) {
                // 过滤空串和停词
                if (word.length() > 0 && !stopWordSet.contains(word)) {
                    outputKey.set(word);
                    outputValue.set(fileName + ",1");
                    context.write(outputKey, outputValue);
                }
            }
        }
    }

    // Reducer阶段：按单词聚合计数，拼接要求格式
    public static class IndexReducer extends Reducer<Text, Text, Text, Text> {
        private final Text result = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            HashMap<String, Integer> docCountMap = new HashMap<>();
            int totalCount = 0;

            // 统计每个文档的词频
            for (Text val : values) {
                String[] split = val.toString().split(",");
                String docName = split[0];
                int count = Integer.parseInt(split[1]);
                docCountMap.put(docName, docCountMap.getOrDefault(docName, 0) + count);
                totalCount += count;
            }

            // 拼接成实验要求的输出格式
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, Integer> entry : docCountMap.entrySet()) {
                sb.append("<").append(entry.getKey()).append(",").append(entry.getValue()).append(">;");
            }
            sb.append("<total,").append(totalCount).append(">.");

            result.set(sb.toString());
            context.write(key, result);
        }
    }

    // 主程序入口
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "InvertedIndex");

        job.setJarByClass(InvertedIndex.class);
        job.setMapperClass(IndexMapper.class);
        job.setReducerClass(IndexReducer.class);
        job.setCombinerClass(IndexReducer.class); // 本地预聚合优化

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // 加载停词表到分布式缓存
        job.addCacheFile(new Path(args[2]).toUri());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
