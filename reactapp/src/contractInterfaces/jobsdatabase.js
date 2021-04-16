import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x2e7E714D77c591F0E930Bb03432Eafd129227220';

export default new web3.eth.Contract(jobs, address);