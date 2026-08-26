import { zodResolver } from '@hookform/resolvers/zod';
import { Loader2, XCircle } from 'lucide-react';
import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';
import { useServerHealth } from '@/lib/hooks/useServer';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';

const connectionSchema = z.object({
  serverUrl: z.string().url('请输入有效的网址'),
});

type ConnectionFormValues = z.infer<typeof connectionSchema>;

export function ConnectionForm() {
  const platform = usePlatform();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const setServerUrl = useServerStore((state) => state.setServerUrl);
  const keepServerRunningOnClose = useServerStore((state) => state.keepServerRunningOnClose);
  const setKeepServerRunningOnClose = useServerStore((state) => state.setKeepServerRunningOnClose);
  const mode = useServerStore((state) => state.mode);
  const setMode = useServerStore((state) => state.setMode);
  const { toast } = useToast();
  const { data: health, isLoading, error: healthError } = useServerHealth();

  const form = useForm<ConnectionFormValues>({
    resolver: zodResolver(connectionSchema),
    defaultValues: {
      serverUrl: serverUrl,
    },
  });

  // Sync form with store when serverUrl changes externally
  useEffect(() => {
    form.reset({ serverUrl });
  }, [serverUrl, form]);

  const { isDirty } = form.formState;

  function onSubmit(data: ConnectionFormValues) {
    setServerUrl(data.serverUrl);
    form.reset(data);
    toast({
      title: '服务器地址已更新',
      description: `已连接到 ${data.serverUrl}`,
    });
  }

  return (
    <Card role="region" aria-label="服务器连接" tabIndex={0}>
      <CardHeader>
        <CardTitle>服务器连接</CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="serverUrl"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>服务器地址</FormLabel>
                  <FormControl>
                    <Input placeholder="http://127.0.0.1:17493" {...field} />
                  </FormControl>
                  <FormDescription>请输入 Voicebox 后端服务器地址</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {isDirty && <Button type="submit">更新连接</Button>}
          </form>
        </Form>

        {/* Connection status */}
        <div className="mt-4">
          {isLoading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm text-muted-foreground">正在检查连接…</span>
            </div>
          ) : healthError ? (
            <div className="flex items-center gap-2">
              <XCircle className="h-4 w-4 text-destructive" />
              <span className="text-sm text-destructive">连接失败：{healthError.message}</span>
            </div>
          ) : health ? (
            <div className="flex flex-wrap gap-2">
              <Badge
                variant={health.model_loaded || health.model_downloaded ? 'default' : 'secondary'}
              >
                {health.model_loaded || health.model_downloaded ? '模型已就绪' : '暂无模型'}
              </Badge>
              <Badge variant={health.gpu_available ? 'default' : 'secondary'}>
                GPU：{health.gpu_available ? '可用' : '不可用'}
              </Badge>
              {health.vram_used_mb != null && health.vram_used_mb > 0 && (
                <Badge variant="outline">VRAM: {health.vram_used_mb.toFixed(0)} MB</Badge>
              )}
            </div>
          ) : null}
        </div>

        <div className="mt-6 pt-6 border-t">
          <div className="flex items-start space-x-3">
            <Checkbox
              id="keepServerRunning"
              className="mt-[6px]"
              checked={keepServerRunningOnClose}
              onCheckedChange={(checked: boolean) => {
                setKeepServerRunningOnClose(checked);
                platform.lifecycle.setKeepServerRunning(checked).catch((error) => {
                  console.error('Failed to sync setting to Rust:', error);
                });
                toast({
                  title: '设置已更新',
                  description: checked
                    ? '关闭应用后服务器将继续运行'
                    : '关闭应用后服务器将停止运行',
                });
              }}
            />
            <div className="space-y-1">
              <label
                htmlFor="keepServerRunning"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
              >
                关闭应用时保持服务器运行
              </label>
              <p className="text-sm text-muted-foreground">
                开启后，关闭应用时服务器仍将在后台运行。此项默认关闭。
              </p>
            </div>
          </div>
        </div>

        {platform.metadata.isTauri && (
          <div className="mt-6 pt-6 border-t">
            <div className="flex items-start space-x-3">
              <Checkbox
                id="allowNetworkAccess"
                className="mt-[6px]"
                checked={mode === 'remote'}
                onCheckedChange={(checked: boolean) => {
                  setMode(checked ? 'remote' : 'local');
                  toast({
                    title: '设置已更新',
                    description: checked
                      ? '已允许网络访问，重启应用后生效。'
                      : '已禁止网络访问，重启应用后生效。',
                  });
                }}
              />
              <div className="space-y-1">
                <label
                  htmlFor="allowNetworkAccess"
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                >
                  允许网络访问
                </label>
                <p className="text-sm text-muted-foreground">
                  允许同一网络中的其他设备访问服务器。修改此设置后请重启应用。
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
