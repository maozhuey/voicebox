import { Loader2, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useServerHealth } from '@/lib/hooks/useServer';
import { useServerStore } from '@/stores/serverStore';

export function ServerStatus() {
  const { data: health, isLoading, error } = useServerHealth();
  const serverUrl = useServerStore((state) => state.serverUrl);

  return (
    <Card role="region" aria-label="服务器状态" tabIndex={0}>
      <CardHeader>
        <CardTitle>服务器状态</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="text-sm text-muted-foreground mb-1">服务器地址</div>
          <div className="font-mono text-sm">{serverUrl}</div>
        </div>

        {isLoading ? (
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">正在检查连接…</span>
          </div>
        ) : error ? (
          <div className="flex items-center gap-2">
            <XCircle className="h-4 w-4 text-destructive" />
            <span className="text-sm text-destructive">连接失败：{error.message}</span>
          </div>
        ) : health ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm">已连接</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge
                variant={health.model_loaded || health.model_downloaded ? 'default' : 'secondary'}
              >
                {health.model_loaded || health.model_downloaded ? '模型已就绪' : '暂无模型'}
              </Badge>
              <Badge variant={health.gpu_available ? 'default' : 'secondary'}>
                GPU：{health.gpu_available ? '可用' : '不可用'}
              </Badge>
              {health.vram_used_mb && (
                <Badge variant="outline">VRAM: {health.vram_used_mb.toFixed(0)} MB</Badge>
              )}
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
