export function getValueById(id: string) {
  const element = document.getElementById(id) as HTMLInputElement;
  if (!element) return null;
  return element.value ? element.value : null;
}
