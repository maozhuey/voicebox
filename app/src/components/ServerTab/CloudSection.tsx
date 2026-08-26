import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Cloud, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { SettingRow, SettingSection } from './SettingRow';

// "Log in with browser" device pairing. The backend opens the system browser
// and completes the code exchange; here we just kick it off and poll status
// until the link goes live. The API key never touches the frontend.
export function CloudSection() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [polling, setPolling] = useState(false);

  const { data: status } = useQuery({
    queryKey: ['cloud-status'],
    queryFn: () => apiClient.getCloudStatus(),
    refetchInterval: polling ? 2000 : false,
  });

  const connected = status?.connected ?? false;

  // Once the browser flow completes, stop polling and celebrate.
  useEffect(() => {
    if (connected && polling) {
      setPolling(false);
      toast({
        title: '已连接 Voicebox 云服务',
        description: `已关联设备：${status?.device_name ?? '当前设备'}。`,
      });
    }
  }, [connected, polling, status?.device_name, toast]);

  // Give up after two minutes so an abandoned browser flow doesn't leave the
  // button stuck on "Waiting for browser…". The backend state stays valid for
  // ten, so the user can simply start again.
  useEffect(() => {
    if (!polling) return;
    const timeoutId = window.setTimeout(() => {
      setPolling(false);
      toast({
        title: '登录超时',
        description: '浏览器登录尚未完成，请重试。',
        variant: 'destructive',
      });
    }, 120_000);
    return () => window.clearTimeout(timeoutId);
  }, [polling, toast]);

  const startLogin = useMutation({
    mutationFn: () => apiClient.startCloudLogin(),
    onSuccess: () => {
      setPolling(true);
      toast({
        title: '请在浏览器中继续',
        description: '授权此设备后返回本页面。',
      });
    },
    onError: (error: Error) =>
      toast({
        title: '无法开始登录',
        description: error.message,
        variant: 'destructive',
      }),
  });

  const disconnect = useMutation({
    mutationFn: () => apiClient.disconnectCloud(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cloud-status'] });
      toast({
        title: '已断开连接',
        description: '当前设备已取消关联；密钥在账户中撤销前仍然有效。',
      });
    },
    onError: (error: Error) =>
      toast({ title: '无法断开连接', description: error.message, variant: 'destructive' }),
  });

  const busy = startLogin.isPending || polling;

  return (
    <SettingSection title="Voicebox Cloud" description="在您的设备之间进行端到端加密备份与同步。">
      <SettingRow
        title={connected ? '已连接' : '账户'}
        description={
          connected
            ? `已关联为 ${status?.device_name ?? '当前设备'}${
                status?.key_prefix ? ` · ${status.key_prefix}…` : ''
              }`
            : '登录后可备份并同步捕获内容和生成记录。'
        }
        action={
          connected ? (
            <Button
              disabled={disconnect.isPending}
              onClick={() => disconnect.mutate()}
              size="sm"
              variant="outline"
            >
              {disconnect.isPending ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  正在断开…
                </>
              ) : (
                '断开连接'
              )}
            </Button>
          ) : (
            <Button disabled={busy} onClick={() => startLogin.mutate()} size="sm">
              {busy ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  {polling ? '正在等待浏览器授权…' : '正在打开…'}
                </>
              ) : (
                <>
                  <Cloud className="h-3.5 w-3.5 mr-1.5" />
                  使用浏览器登录
                </>
              )}
            </Button>
          )
        }
      />

      {connected && (
        <SettingRow title="管理" description="可在账户中撤销此设备、添加 API 密钥或管理账单。">
          <a
            className="text-sm text-accent hover:underline"
            href={status?.dashboard_url ?? 'https://voicebox.sh/account'}
            rel="noopener noreferrer"
            target="_blank"
          >
            打开账户管理页面 ↗
          </a>
        </SettingRow>
      )}
    </SettingSection>
  );
}
